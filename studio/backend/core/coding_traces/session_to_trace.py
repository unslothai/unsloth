# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Builds a coding trace from one saved Studio chat thread, in the order the UI shows.

The system text uses the thread's latest saved settings and the project's current instructions;
Studio keeps no history of either, so an earlier turn may have seen different text.
"""

from __future__ import annotations

import json
import math
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
from core.inference.media_chat_tools import MEDIA_CHAT_TOOLS
from core.inference.tool_loop_controller import (
    TOOL_MODEL_SUFFIX_KEY,
    restore_replayed_tool_results,
)
from core.rag.conversation_archive import (
    _flushes_local_pair,
    _local_round_id,
    _sandbox_wrapper,
    _sanitised_assistant_text,
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

# fromThreadMessageLike in @assistant-ui/core keeps only these assistant images.
# JS "." stops at line breaks, so a data URL with one is dropped too.
_DATA_IMAGE = re.compile(r"data:image/(png|jpeg|jpg|gif|webp);base64,[^\n\r\u2028\u2029]*")

# The characters JS trim() removes; Python's str.strip() also strips \x1c-\x1f and \x85, not \ufeff.
_JS_SPACE = (
    " \t\n\v\f\r\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)


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
    """Parts minus mcpToolOnly ones (modelVisibleMessage); only user rows keep attachments."""
    parts = message.get("content")
    if message.get("role") == "user":
        # By choice, [] (nothing typed, or a restored backup) takes the live first send: only the
        # attachments. A reload, and so every restored [] row, sends a "\n" before their text
        # instead (toThreadMessage loads "" and chat-adapter.ts joins with "\n"). Content that is
        # not a list never comes from a send, so it keeps that replay form.
        loaded = parts if isinstance(parts, list) else [{"type": "text", "text": ""}]
        parts = _with_attachments({**message, "content": loaded})
    if not isinstance(parts, list):
        return []  # an assistant's lone empty text part sends nothing
    return [p for p in parts if not (isinstance(p, dict) and p.get("mcpToolOnly") is True)]


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
    return _legacy_load(chain, messages)


def _legacy_load(chain: list[dict], messages: list[dict]) -> list[dict]:
    """With no parentId on any row, the UI loads through fromArray, whose fromThreadMessageLike
    drops blank text and reasoning and non-data-URL images from assistant turns."""
    if any(m.get("parentId") is not None for m in messages):
        return chain  # repository.import keeps every part
    return [
        {**m, "content": [p for p in m["content"] if not _legacy_dropped(p)]}
        if m.get("role") != "user" and isinstance(m.get("content"), list)
        else m
        for m in chain
    ]


def _legacy_dropped(part: object) -> bool:
    kind = part.get("type") if isinstance(part, dict) else None
    if kind in ("text", "reasoning"):
        return not str(part.get("text") or "").strip(_JS_SPACE)
    return kind == "image" and not _DATA_IMAGE.fullmatch(str(part.get("image") or ""))


def _convert(chain: list[dict], warnings: list[TraceWarning]) -> list[TraceMessage]:
    out: list[TraceMessage] = []
    call_ids: set[str] = set()
    for message in chain:
        groups = _convert_row(message, out, call_ids, warnings)
        _clear_stopped_reasoning(message, groups)
    return _replayable(out)


def _convert_row(
    message: dict, out: list[TraceMessage], call_ids: set[str], warnings: list[TraceWarning]
) -> _UiGroups:
    """Append one stored row's trace messages to ``out``; returns the row's UI groups."""
    # toThreadMessage loads every row that is not "user" as an assistant turn.
    role = "user" if message.get("role") == "user" else "assistant"
    message_id, groups = message["id"], _UiGroups()
    if role == "user":
        current = TraceMessage(id = message_id, role = role, created_at = message.get("createdAt"))
        out.append(current)
    # Tool messages wait for their call group to close (serializeAssistantReplayMessages).
    pending: list[TraceMessage] = []
    pending_round = reasoned = imaged = None
    # toOpenAIMessages (user) and the flush (assistant) join text with "\n" and keep empty parts.
    pieces, texted, split = 0, None, False
    for position, part in enumerate(_model_parts(message)):
        kind = part.get("type") if isinstance(part, dict) else None
        if role == "assistant" and kind in ("text", "reasoning", "tool-call"):
            skipped = _skipped_call(part, message_id, position) if kind == "tool-call" else None
            if skipped:
                warnings.append(skipped)
                if _server_builtin(part)[1]:
                    groups.native(out[-1] if out else None, texted)
                    split = True
                continue
            flush_pair, pending_round = _close_call_group(part, kind, pending, pending_round, out)
            # A native call stays pending, so the next text or reasoning starts a new message.
            pieces = _open_assistant(message, out, pieces, split and kind != "tool-call")
            split = split and kind == "tool-call"
            current = groups.enter(out[-1])
        if kind == "text":
            text = str(part.get("text") or "")
            current.text, texted = _append_text(current.text, text, current is texted), current
        elif kind == "reasoning" and role == "assistant":
            reasoning = str(part.get("text") or "")
            current.reasoning = _append_text(current.reasoning, reasoning, current is reasoned)
            reasoned = current  # pendingReasoningParts.join("\n") also keeps "" parts
        elif kind == "tool-call" and role == "assistant":
            call_id = _unique_call_id(part, message_id, position, call_ids, warnings)
            name = str(part.get("toolName") or "")
            current.tool_calls.append(ToolCall(call_id, name or "unknown", _call_arguments(part)))
            pending.append(_tool_message(part, name, call_id, message))
            if flush_pair:
                _flush(pending, out)
                pending_round = None
        else:
            imaged = imaged or (kind == "image" and part.get("image"))
            warnings.append(
                TraceWarning("unsupported_part", f"Dropped a {kind!r} part", message_id)
            )
    out.extend(pending)
    groups.imaged = bool(imaged)
    return groups


def _open_assistant(message: dict, out: list[TraceMessage], pieces: int, split: bool) -> int:
    # Importers split one reply into a row per block; merge rows until a tool result.
    if out and out[-1].role == "assistant" and not split:
        return pieces
    pieces += 1
    piece = message["id"] if pieces == 1 else f"{message['id']}#{pieces}"
    out.append(TraceMessage(piece, "assistant", created_at = message.get("createdAt")))
    return pieces


def _flush(pending: list[TraceMessage], out: list[TraceMessage]) -> None:
    out.extend(pending)
    pending.clear()


def _close_call_group(
    part: dict,
    kind: str,
    pending: list[TraceMessage],
    pending_round: Optional[str],
    out: list[TraceMessage],
) -> tuple[bool, Optional[str]]:
    """Flush waiting tool messages when this part ends their call group; returns
    ``(flush_pair, pending_round)``, where flush_pair means the call's own result flushes next."""
    round_id = _local_round_id(part) if kind == "tool-call" else None
    flush_pair = kind == "tool-call" and round_id is None and _flushes_local_pair(part)
    if pending and (
        kind != "tool-call"
        or flush_pair
        or (round_id is not None and pending_round not in (None, round_id))
    ):
        _flush(pending, out)
        pending_round = None
    return flush_pair, round_id if round_id is not None else pending_round


def _unique_call_id(
    part: dict, message_id: str, position: int, call_ids: set[str], warnings: list[TraceWarning]
) -> str:
    call_id = str(part.get("toolCallId") or f"{message_id}-{position}")
    if call_id in call_ids:
        names = (f"{call_id}#{n}" for n in range(2, len(call_ids) + 3))
        renamed = next(name for name in names if name not in call_ids)
        note = f"Tool call id {call_id!r} repeats; renamed to {renamed!r}"
        warnings.append(TraceWarning("duplicate_tool_call_id", note, message_id))
        call_id = renamed
    call_ids.add(call_id)
    return call_id


def _call_arguments(part: dict) -> dict:
    args, raw = part.get("args"), part.get("argsText")
    try:  # toolCallReplayArguments sends argsText as is when it is one JSON object
        exact = json.loads(raw, parse_float = _finite_float, parse_constant = _reject_constant)
    except (TypeError, ValueError, RecursionError):
        exact = None
    marker = isinstance(raw, str) and raw and args == {"_raw": raw}  # replays as {}
    args = exact if isinstance(exact, dict) else {} if marker else args
    return args if isinstance(args, dict) else {}


def _tool_message(part: dict, name: str, call_id: str, message: dict) -> TraceMessage:
    wire = {"role": "tool", "name": name}
    result = part["result"]  # isMcpUiToolResult
    # SANDBOX_FILE_TOOLS also holds MEDIA_CHAT_TOOLS, which the archive's name set lacks.
    if name in MEDIA_CHAT_TOOLS and _sandbox_wrapper(result, "python"):
        result = result["text"]
    ui = _as_dict(result.get("ui")) if isinstance(result, dict) else {}
    if name.startswith("mcp__") and isinstance(ui.get("resourceUri"), str):
        result = result["text"] if isinstance(result.get("text"), str) else result
    wire["content"] = _tool_result_content(result, name)
    suffix = _as_dict(part.get("provenance")).get("model_suffix")
    if isinstance(suffix, str) and suffix:
        wire["extra_content"] = {TOOL_MODEL_SUFFIX_KEY: suffix}
    return TraceMessage(
        id = f"{message['id']}:{call_id}",
        role = "tool",
        text = restore_replayed_tool_results([wire])[0]["content"],
        tool_call_id = call_id,
        created_at = message.get("createdAt"),
    )


class _UiGroups:
    """One row's UI message groups (serializeAssistantReplayMessages). The trace can merge several
    groups into one message, so each mark is ``[message, kept reasoning, text start, call start,
    first group]``. A stopped turn decides each group's reasoning on its own."""

    def __init__(self) -> None:
        self.marks: dict[int, list] = {}
        self.closed = False  # a native call closed a group before the row had a message
        self.imaged = False

    def enter(self, m: TraceMessage) -> TraceMessage:
        if id(m) not in self.marks:
            self._mark(m, not (self.marks or self.closed), False)
        return m

    def _mark(self, m: TraceMessage, first: bool, joined: bool) -> None:
        start = len(m.text) + (bool(m.text) or joined)  # skip the "\n" _append_text joins with
        self.marks[id(m)] = [m, len(m.reasoning), start, len(m.tool_calls), first]

    def native(self, m: Optional[TraceMessage], texted: Optional[TraceMessage]) -> None:
        # A native call ends its group, and the call keeps that group's reasoning.
        if m is not None and id(m) in self.marks:
            self._mark(m, False, m is texted)
        else:
            self.closed = True

    def stop(self) -> None:
        for m, kept, text_start, call_start, first in self.marks.values():
            # shouldReplayAssistantReasoning: the text check is untrimmed; images go to the
            # first group (collectImageParts).
            text = _sanitised_assistant_text(m.text[text_start:])
            if text or len(m.tool_calls) > call_start or (first and self.imaged):
                kept = len(m.reasoning)
            m.reasoning = m.reasoning[:kept]


def _clear_stopped_reasoning(message: dict, groups: _UiGroups) -> None:
    reason = _as_dict(_as_dict(message.get("metadata")).get("incomplete")).get("reason")
    if isinstance(reason, str) and reason in _STOP_LABELS:
        groups.stop()


def _replayable(out: list[TraceMessage]) -> list[TraceMessage]:
    for m in out:
        if m.role == "assistant":
            m.text = _sanitised_assistant_text(m.text)
    # toOpenAIMessages sends every user turn, even an empty one.
    return [
        m
        for m in out
        # The UI sends a group whose text is not empty, without trimming it.
        if m.role in ("user", "tool") or m.text or m.reasoning or m.tool_calls
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


def _replayed(message: dict, reasoning: bool) -> bool:
    # The UI sends an image (collectImageParts) and a native Gemini call; the trace drops both.
    return any(
        p.get("image") if p.get("type") == "image" else _server_builtin(p)[1]
        for p in _model_parts(message)
        if isinstance(p, dict) and p.get("type") in ("image", "tool-call")
    ) or any(
        m.text.strip(_JS_SPACE) or m.tool_calls or (m.reasoning and (reasoning or m.text))
        for m in _convert([message], [])
    )


def _abandoned(message: dict) -> bool:
    # assistantTurnCarriesPayload reads the raw parts, before the replay sanitiser, and
    # isAbandonedAssistantTurn keeps any turn with a reasoning part, even an empty one.
    payload = any(
        isinstance(p, dict)
        and (
            p.get("type") == "reasoning"
            or (p.get("type") == "image" and p.get("image"))
            or (p.get("type") == "text" and str(p.get("text") or "").strip(_JS_SPACE))
        )
        for p in _model_parts(message)
    )
    return not payload and not _replayed(message, reasoning = True)


def _prune(chain: list[dict], warnings: list[TraceWarning]) -> list[dict]:
    """Drop refused and abandoned turns as pruneOutboundHistory (chat-adapter.ts) does, and give an
    empty stopped turn its label, as fillStoppedAssistantReplay does. An unsettled run's row gets no
    "Response interrupted" label: the UI asks the server if the run is live, and rows lack that.
    Each part an abandoned or labelled turn drops, and the prompt an abandoned turn takes with it,
    gets the warning it would get in a kept turn. A refused turn and its prompt get none: the UI
    drops that exchange by design, not because a run failed, so the trace loses nothing it needs."""

    meta = [_as_dict(m.get("metadata")) for m in chain]  # runtime-provider.tsx: metadata.custom
    assistant = [m.get("role") != "user" for m in chain]
    refused = [a and d.get("anthropicRefusal") is True for d, a in zip(meta, assistant)]
    reasons = [_as_dict(d.get("incomplete")).get("reason") for d in meta]
    stopped = [isinstance(r, str) and r in _STOP_LABELS for r in reasons]
    dropped = [
        r or (a and not s and _abandoned(m))
        for m, a, r, s in zip(chain, assistant, refused, stopped)
    ]
    last = max((i for i, d in enumerate(dropped) if not d), default = -1)
    kept: list[dict] = []
    for index, message in enumerate(chain):
        if dropped[index]:
            if not refused[index]:
                # An abandoned turn replays nothing, so this adds only the warnings it would get.
                _convert([message], warnings)
            if (refused[index] or index < last) and kept and kept[-1].get("role") == "user":
                if not refused[index]:
                    _convert([kept[-1]], warnings)  # its prompt goes with it
                kept.pop()
            continue
        # The UI never replays the reasoning of a stopped turn with no text and no calls.
        if assistant[index] and stopped[index] and not _replayed(message, reasoning = False):
            label = _STOP_LABELS[reasons[index]]
            _convert([message], warnings)  # the label replaces parts that would warn if kept
            message = {**message, "content": [{"type": "text", "text": label}], "attachments": []}
        kept.append(message)
    return kept


def _js_number(value: float) -> str:
    """Number.prototype.toString for a double, from Python's shortest round-trip digits."""
    if not math.isfinite(value):
        return ("-" if value < 0 else "") + "Infinity"  # JSON.parse gives no NaN
    if value == 0:
        return "0"
    mantissa, _, exponent = repr(abs(value)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction.rstrip("0")).lstrip("0")
    point = len(whole.lstrip("0")) + int(exponent or 0)  # where the decimal point falls
    if not whole.lstrip("0"):
        point -= len(fraction) - len(fraction.lstrip("0"))
    sign, k = "-" if value < 0 else "", len(digits)
    if k <= point <= 21:
        return sign + digits + "0" * (point - k)
    if 0 < point <= 21:
        return sign + digits[:point] + "." + digits[point:]
    if -6 < point <= 0:
        return sign + "0." + "0" * -point + digits
    e = point - 1
    tail = f"e{'+' if e > 0 else '-'}{abs(e)}"
    return sign + digits[0] + ("." + digits[1:] if k > 1 else "") + tail


def _js_string(text: str) -> str:
    # JSON.stringify escapes a lone surrogate; json.loads already joins the valid pairs.
    quoted = json.dumps(text, ensure_ascii = False)
    return re.sub(r"[\ud800-\udfff]", lambda c: f"\\u{ord(c.group()):04x}", quoted)


def _array_index(key: str) -> bool:
    return key.isascii() and key.isdigit() and str(int(key)) == key and int(key) < 2**32 - 1


def _js_json(value: object) -> str:
    """JSON.stringify for a JSON.parse result."""
    if isinstance(value, dict):
        # Object keys that are array indexes come first, in ascending order.
        indexes = sorted((k for k in value if _array_index(k)), key = int)
        keys = indexes + [k for k in value if not _array_index(k)]
        return "{" + ",".join(f"{_js_string(k)}:{_js_json(value[k])}" for k in keys) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_js_json(v) for v in value) + "]"
    if isinstance(value, float):
        return _js_number(value) if math.isfinite(value) else "null"
    return _js_string(value) if isinstance(value, str) else json.dumps(value)


def _finite_float(text: str) -> float:
    # JS parses 1e400 as Infinity, which a dict cannot carry as valid JSON; use args instead.
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite number")
    return value


def _reject_constant(text: str) -> None:
    raise ValueError(f"{text} is not JSON")


def _fill_variables(prompt: str, raw: object) -> str:
    """Fill custom systemVariables as resolveSystemPromptVariables does; $date, $time and $now
    used the send-time clock, which is not stored, so they stay as written."""
    try:
        # parseSystemVariablesMap uses JSON.parse, so every number is a double and a NaN
        # fills nothing.
        values = (
            json.loads(raw, parse_int = float, parse_constant = _reject_constant)
            if isinstance(raw, str) and raw.strip(_JS_SPACE)
            else {}
        )
    except (ValueError, RecursionError):
        values = {}

    def fill(match: re.Match) -> str:
        current = values if isinstance(values, dict) else {}
        if match.group(1) in ("$date", "$time", "$now"):
            return match.group(0)
        for key in filter(None, (p.strip(_JS_SPACE) for p in match.group(1).split("."))):
            if not isinstance(current, dict) or key not in current:
                return match.group(0)
            current = current[key]
        if current is None:
            return ""
        if isinstance(current, str):
            return current
        if isinstance(current, float):
            return _js_number(current)  # String(); JSON.stringify would print Infinity as null
        try:
            return _js_json(current)
        except RecursionError:
            return match.group(0)  # too deep to print here; JSON.stringify would manage it

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
        instructions = str(project.get("instructions") or "").strip(_JS_SPACE)
    if agents_md_enabled():
        note = "AGENTS.md is read at send time and not stored; it may have come before this text"
        warnings.append(TraceWarning("agents_md_not_stored", note))
    if instructions:
        instructions = f"<project_instructions>\n{instructions}\n</project_instructions>"
    text = "\n\n".join(part for part in (instructions, prompt.strip(_JS_SPACE)) if part)
    if not text:
        return None
    return TraceMessage(id = f"{thread['id']}:system", role = "system", text = text)


def session_to_trace(
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
    chain = _branch(messages, head_id, warnings)
    row_warnings: list[TraceWarning] = []
    trace_messages = _convert(_prune(chain, row_warnings), row_warnings)
    # _prune warns for the turns it drops before _convert runs; a stable sort restores row order.
    order = {m["id"]: i for i, m in enumerate(chain)}
    warnings += sorted(row_warnings, key = lambda w: order[w.message_id])
    if not any(m.role != "user" or m.text for m in trace_messages):
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


def load_session_trace(thread_id: str, *, head_id: Optional[str] = None) -> ChatTrace:
    """Load a saved Studio thread from the database and turn it into a coding trace; head_id
    works as in session_to_trace."""
    thread = studio_db.get_chat_thread(thread_id)
    if thread is None:
        raise ChatTraceError(f"Thread {thread_id!r} does not exist")
    project_id = thread.get("projectId")
    project = studio_db.get_chat_project(project_id) if project_id else None
    return session_to_trace(
        thread, studio_db.list_chat_messages(thread_id), project = project, head_id = head_id
    )
