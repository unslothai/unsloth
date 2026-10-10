# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth-owned tool execution loop, shared by every external provider transport.

The loop owns the parts that do not depend on how bytes reach the provider: turn cycling, the tool
budget, the approval handshake, local execution through ``execute_tool``, and the conversation
replay that carries results back. A ``ToolLoopTransport`` supplies the one thing that does differ,
an async iterator of OpenAI-shaped SSE lines for one turn. Two transports exist: ``CodexTransport``
(the ChatGPT subscription, which speaks its own app-server protocol) and ``OAICompatTransport``
(everything routed through ``ExternalProviderClient``, which already normalises Anthropic, Gemini
and the Responses API into OpenAI chunk shape).

Self-hosted models frequently write tool calls as text rather than emitting structured
``delta.tool_calls``. A transport that sets ``heals_text_tool_calls`` gets the content stream routed
through ``StreamToolCallHealer``, the same bounded buffer the client-tool passthrough uses: only a
trailing partial-signal window or a suspected tool block is ever withheld, and a block that cannot
become a declared call flushes verbatim. Nothing here invents a cap on how much model output may be
held.

Origins. This file is assembled out of four contributor PRs rather than written from scratch, and
the squash merge of #8665 did not carry their authorship, so it is recorded here. #8626 (khalidejaz)
contributed the registry capability and the hidden-entry exposure, including the
``provider_runs_local_tools()`` name ``providers.py`` still uses. #8630 (mahiatlinux) contributed
most of the loop internals below: the user-turn append, the tool-stream advance and drain, usage
merging, the approval flush delay, the budget-exhausted nudge, and the streamed tool-name rule that
llama-server forced, plus the browser testing against a control worktree that caught a truncated
turn discarding a healed call along with the text describing it. #7805 (Etherll) contributed the
hosted-provider reach, and #7330 (Souravrajvi0) the original OpenAI-compatible framing.

Two ideas from those PRs were deliberately not taken, which is worth knowing before anyone
re-derives them: the per-connection opt-in from #8630 and the ``studio_tool_execution`` column from
#7805. #8665 shipped a static disclosure instead, while widening the capability from four
self-hosted types to thirteen.
"""

from __future__ import annotations

import asyncio
import json
import threading

from dataclasses import dataclass, field
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from core.inference import tools as tools_module
from core.inference.chat_template_helpers import append_assistant_turn
from core.inference.passthrough_healing import StreamToolCallHealer, heal_gate, nudge_enabled
from core.inference.sse_control_frames import sanitize_provider_sse_line
from core.inference.tool_call_parser import (
    MAX_ACT_REPROMPTS,
    is_reprompt_repeat,
    is_short_intent_without_action,
    reprompt_to_act_message,
    strip_tool_markup,
)
from core.inference.mcp_images import append_image_turn as append_mcp_image_turn
from core.inference.mcp_image import note_attached_image


def _append_mcp_images_owned(
    conversation,
    results,
    owned,
    lead = None,
):
    """reserve caller images because remote providers apply the image cap in document order."""
    append_mcp_image_turn(
        conversation,
        results,
        per_result = True,
        owned = owned,
        reserve_caller_images = True,
        **({"lead": lead} if lead else {}),
    )


from core.inference.tool_loop_controller import (
    ToolLoopController,
    _reject_json_constant,
    awaiting_approval_status,
    canonical_arguments_text,
    mcp_display_parts,
    provisional_tool_provenance,
    strip_result_for_model,
)
from core.inference.tool_stream_exec import (
    TOOL_HEARTBEAT_INTERVAL_S,
    accepts_kwarg,
    accepts_output_callback,
    search_images_kwargs,
    stream_tool_execution,
)
from core.inference.tools import (
    build_rag_autoinject,
    execute_tool,
    is_high_risk_tool_call,
    mcp_image_share,
    mcp_image_targets,
    never_needs_approval,
)
from state.tool_approvals import (
    DECISION_EXPIRED,
    TOOL_APPROVAL_EXPIRED_MESSAGE,
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    decision_reason,
    new_approval_id,
    wait_tool_decision,
)


_TOOL_BUDGET_EXHAUSTED = (
    "Unsloth did not execute this tool call because the per-message tool-call limit was reached. "
    "Continue with the available results and answer without calling another tool."
)

_TOOL_DISABLED = "Unsloth did not execute this tool call because the tool is disabled."

_TOOL_CANCELLED = (
    "Unsloth stopped this tool call before it returned, so there is no result. "
    "The tool may have already done part of its work."
)

_TOOL_TRUNCATED = (
    "Unsloth did not execute this tool call because the provider stopped mid-call at its "
    "output limit."
)

_TOOL_CHOICE_NONE = (
    "Unsloth did not execute this tool call because tool calls are turned off for this request "
    '(tool_choice is "none").'
)

# Card text for a call the controller skipped. The client already painted a card from the provider's own tool_calls
# delta, so it needs a short result; the long model-facing nudge stays in the conversation.
_TOOL_SKIPPED = {
    "duplicate": "Unsloth did not run this call because an identical one had already completed.",
    "disabled": _TOOL_DISABLED,
    "render_html_repeat": "Unsloth did not run this call because render_html already ran.",
}

_BUDGET_EXHAUSTED_NUDGE = (
    "You have used all available tool calls. Based on everything you have found "
    "so far, provide your final answer now. Do not call any more tools."
)

_SSE_KEEPALIVE = ": keep-alive"

# Separate write so a desktop webview paints Allow / Deny before the tool blocks.
_TOOL_APPROVAL_FLUSH_DELAY_S = 0.05

_MAX_POST_TOOL_REPROMPTS = 1

_USAGE_DETAIL_FIELDS = (
    "prompt_tokens_details",
    "completion_tokens_details",
    "cache_creation",
)

_STEP_DONE = object()


def _truncate_for_model(
    text: str,
    limit: int | None = None,
    *,
    joiner: str = "\n",
) -> str:
    """Hold a hosted result to the same cap a local result gets. Read off ``tools`` rather than
    copied, so an install that lowers ``UNSLOTH_TOOL_RESULT_MAX_CHARS`` gets the lower cap here
    too."""
    if limit is None:
        limit = tools_module._MAX_OUTPUT_CHARS
    if len(text) <= limit:
        return text
    return text[:limit] + f"{joiner}... [truncated, {len(text) - limit} more characters]"


_HOSTED_ARGUMENT_MAX_CHARS = 2000


# Provider replay plumbing (Gemini executableCode/thoughtSignature, OpenAI reasoning items),
# never meant for the model.
_HOSTED_ARGUMENT_PLUMBING_KEYS = frozenset({"google", "_server_tool"})


def _carries_image_sentinel(result: str) -> bool:
    """Whether a hosted result ends in the ``__IMAGES__`` envelope itself. Validated rather than
    matched on sight, the way the local strippers validate theirs: a fetched page that merely
    writes the marker is prose, and reading it as a picture would report an image the turn never
    made."""
    _, sep, payload = result.rpartition("\n__IMAGES__:")
    if not sep:
        return False
    try:
        images = json.loads(payload)
    except (ValueError, RecursionError):
        return False
    return (
        isinstance(images, list) and bool(images) and all(isinstance(i, str) and i for i in images)
    )


def _hosted_arguments_for_model(arguments: Any) -> dict[str, Any]:
    """The part of a hosted tool's arguments worth showing the model."""
    if not isinstance(arguments, dict):
        return {}
    return {
        key: value
        for key, value in arguments.items()
        if key not in _HOSTED_ARGUMENT_PLUMBING_KEYS and not key.startswith("openai_")
    }


_MAX_FRUITLESS_TURNS = 2


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, separators = (",", ":"))


def _is_done_sentinel(line: str) -> bool:
    return line.startswith("data:") and line[5:].strip() == "[DONE]"


def _chunk_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    raw = line[5:].strip()
    if not raw or raw == "[DONE]":
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _mint_streamed_card_id(taken: set[str], index: Any) -> str:
    """The id the client painted an id-less call's card under. ``tool_call_<delta index>``, else the
    lowest free ``tool_call_<n>``: the client's rule, reproduced so tool_start and tool_end reach
    the card the stream drew. A card id only; the replayed id stays ``call_<round>_<pos>``."""
    preferred = (
        f"tool_call_{index}" if isinstance(index, int) and not isinstance(index, bool) else ""
    )
    if preferred and preferred not in taken:
        return preferred
    position = 0
    while f"tool_call_{position}" in taken:
        position += 1
    return f"tool_call_{position}"


def _normalized_call(call: dict[str, Any], fallback_id: str = "") -> dict[str, Any] | None:
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    if not isinstance(call_id, str) or not call_id:
        # Some OpenAI-compatible servers omit the id; mint one rather than drop the call.
        call_id = fallback_id
    if not call_id:
        return None
    name = function.get("name")
    arguments = function.get("arguments", "")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = {"_raw": arguments}
    except RecursionError:
        # Deeply nested but valid JSON blows the interpreter's stack rather than failing to decode
        parsed = {"_raw": arguments}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    normalized: dict[str, Any] = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments or "{}"},
        "arguments": parsed,
    }
    extra = call.get("extra_content")
    if isinstance(extra, dict) and extra:
        normalized["extra_content"] = extra
    return normalized


def _signed_provider_call_for_replay(call: dict[str, Any]) -> dict[str, Any] | None:
    extra = call.get("extra_content")
    google = extra.get("google") if isinstance(extra, dict) else None
    if not isinstance(google, dict) or not google.get("thought_signature"):
        return None
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    return {
        "id": call["id"],
        "type": "function",
        "function": {
            "name": function["name"],
            "arguments": function.get("arguments", ""),
        },
        "extra_content": extra,
    }


def _argument_fragment(value: Any) -> Any:
    """A decoded-object ``arguments`` delta as the text it would have streamed as. llama-server has
    shipped a decoded object where a string fragment belongs (ggml-org/llama.cpp#20198).
    Everything else passes through untouched, so ``None`` still means "announced, no arguments
    field". Mirrors ``streamedToolCallArguments`` in ``tool-call-arguments.ts``."""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii = False, separators = (",", ":"))
        except (TypeError, ValueError, RecursionError):
            return ""
    return value


def _delta_text(content: Any) -> str:
    """Text of a content delta, whether it is a plain string or content parts. Structured content
    blocks reach the client fine either way, but only the text of them belongs in the assistant
    message replayed upstream, and dropping it there loses the turn's prose on the follow-up
    call."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def _tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    return {
        name
        for tool in tools or []
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and isinstance((name := tool["function"].get("name")), str)
        and name
    }


class ToolLoopTransport(Protocol):
    """One turn of provider inference, as OpenAI-shaped SSE lines."""

    heals_text_tool_calls: bool

    # True if the transport already stripped control vocabulary; sanitizing twice drops its frames.
    sanitizes_provider_frames: bool

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        cancel_event: threading.Event,
    ) -> AsyncIterator[str]: ...


@dataclass(frozen = True)
class ToolLoopRun:
    """Everything about the request that the loop, not the transport, needs."""

    messages: list[dict[str, Any]]
    session_id: str | None = None
    thread_id: str | None = None
    model: str | None = None
    tool_choice: Any = None
    continue_final_message: bool = False
    supports_vision: bool = False
    # Counted toward the loop's image cap so resumed chats cannot exceed it.
    promoted_image_parts: tuple = ()


@dataclass(frozen = True)
class ToolLoopPolicy:
    tools: list[dict[str, Any]]
    max_calls: int
    timeout: int
    permission_mode: str
    confirm_calls: bool
    bypass_permissions: bool
    rag_scope: dict[str, Any] | None
    auto_heal: bool | None = None
    nudge_tool_calls: bool | None = None
    on_withheld_tool_call: Callable[[], None] | None = None
    # Headerless only: a turn may end on [DONE] alone, so the wire cannot always clear the flag.
    on_provider_turn_end: Callable[[], None] | None = None
    # None keeps the request default (on); explicit booleans win.
    deduplicate_tool_calls: bool | None = None
    sandbox_level: str = "high"


def _split_top_level_json_objects(text: str) -> tuple[list[str], str]:
    """The top-level JSON objects in ``text``, and any object still unfinished. A second top-level
    ``{`` means one slot took a second parallel call. Text that is not a run of whole objects
    comes back whole, so a stream this was never meant for is left alone. Must agree with
    ``splitTopLevelJsonObjects`` in ``chat/tool-call-arguments.ts``."""
    unsplit: tuple[list[str], str] = ([], text)
    complete: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if depth == 0:
            if ch == "{":
                depth = 1
                start = i
                continue
            if ch in " \t\n\r":
                continue
            return unsplit
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                segment = text[start : i + 1]
                try:
                    json.loads(
                        segment,
                        parse_constant = _reject_json_constant,
                        # As text: JSON.parse has no 4300-digit int cap.
                        parse_int = str,
                    )
                except (ValueError, TypeError):
                    return unsplit
                except RecursionError:
                    return unsplit
                complete.append(segment)
                start = -1

    return complete, ("" if start == -1 else text[start:])


@dataclass
class _BoundaryScan:
    """``_split_top_level_json_objects`` over a string that only ever grows. Rescanning per fragment
    is O(N^2); resuming is one pass. ``feed`` takes the same string extended, never a rewritten
    one, so ``_Turn`` drops the scan for any key a fork rewrites."""

    depth: int = 0
    start: int = -1
    in_string: bool = False
    escaped: bool = False
    scanned: int = 0
    complete: list[str] = field(default_factory = list)
    unsplittable: bool = False

    def feed(self, text: str) -> tuple[list[str], str]:
        if self.unsplittable:
            return [], text
        i = self.scanned
        while i < len(text):
            ch = text[i]
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif ch == "\\":
                    self.escaped = True
                elif ch == '"':
                    self.in_string = False
                i += 1
                continue
            if self.depth == 0:
                if ch == "{":
                    self.depth = 1
                    self.start = i
                    i += 1
                    continue
                if ch in " \t\n\r":
                    i += 1
                    continue
                self.unsplittable = True
                return [], text
            if ch == '"':
                self.in_string = True
            elif ch == "{":
                self.depth += 1
            elif ch == "}":
                self.depth -= 1
                if self.depth == 0:
                    segment = text[self.start : i + 1]
                    try:
                        json.loads(
                            segment,
                            parse_constant = _reject_json_constant,
                            parse_int = str,
                        )
                    except (ValueError, TypeError, RecursionError):
                        self.unsplittable = True
                        return [], text
                    self.complete.append(segment)
                    self.start = -1
            i += 1
        self.scanned = len(text)
        return list(self.complete), ("" if self.start == -1 else text[self.start :])


@dataclass
class _Turn:
    """Accumulated state for one provider turn."""

    by_index: dict[Any, dict[str, Any]] = field(default_factory = dict)
    order: list[Any] = field(default_factory = list)
    # delta index -> call key: index, then (index, call_id) after a fork, or (index, "_split", n).
    open_key_by_index: dict[int, Any] = field(default_factory = dict)
    last_index: int | None = None
    split_seq: int = 0
    seq_by_key: dict[Any, int] = field(default_factory = dict)
    seq_counter: int = 0
    key_by_call_id: dict[str, Any] = field(default_factory = dict)
    scan_by_key: dict[Any, _BoundaryScan] = field(default_factory = dict)
    # Reported only once closed, so a stream cut after '{"a":1}{' cannot run half an argument.
    open_tail_keys: set[Any] = field(default_factory = set)
    # A repeated name is a resend or the next call; only the following object tells them apart.
    pending_extra: dict[Any, dict[str, Any]] = field(default_factory = dict)
    round: int = 0
    healed: list[dict[str, Any]] = field(default_factory = list)
    text: list[str] = field(default_factory = list)
    reasoning: list[str] = field(default_factory = list)
    reasoning_extra: dict[str, Any] | None = None
    finish_reason: str | None = None
    hosted_results: dict[str, dict[str, Any]] = field(default_factory = dict)
    provider_compaction: dict[str, Any] | None = None

    def note_reasoning_extra(self, extra: Any) -> None:
        """Accumulate Gemini's per-part replay envelopes across one provider turn."""
        if not isinstance(extra, dict):
            return
        previous = self.reasoning_extra if isinstance(self.reasoning_extra, dict) else {}
        merged = {**previous, **extra}
        previous_google = previous.get("google")
        incoming_google = extra.get("google")
        if isinstance(previous_google, dict) or isinstance(incoming_google, dict):
            old_google = previous_google if isinstance(previous_google, dict) else {}
            new_google = incoming_google if isinstance(incoming_google, dict) else {}
            google = {**old_google, **new_google}
            for singular, plural in (
                ("thought_part", "thought_parts"),
                ("answer_part", "answer_parts"),
            ):
                accumulated: list[dict[str, Any]] = []
                for source in (old_google, new_google):
                    many = source.get(plural)
                    if isinstance(many, list):
                        accumulated.extend(dict(part) for part in many if isinstance(part, dict))
                    one = source.get(singular)
                    if isinstance(one, dict):
                        accumulated.append(dict(one))
                if accumulated:
                    compacted: list[dict[str, Any]] = []
                    for part in accumulated:
                        signature = part.get("thought_signature") or part.get("thoughtSignature")
                        if (
                            compacted
                            and not signature
                            and not (
                                compacted[-1].get("thought_signature")
                                or compacted[-1].get("thoughtSignature")
                            )
                            and isinstance(part.get("text"), str)
                            and isinstance(compacted[-1].get("text"), str)
                        ):
                            compacted[-1]["text"] += part["text"]
                        else:
                            compacted.append(part)
                    google[plural] = compacted
                google.pop(singular, None)
            if google.get("thought_parts") or google.get("answer_parts"):
                # Native ledgers already pin every signature to its original part. Retaining the legacy scalar
                # would let the translator move an earlier thought signature onto later unsigned text.
                google.pop("thought_signature", None)
                google.pop("thoughtSignature", None)
                google.pop("thought", None)
            merged["google"] = google
        self.reasoning_extra = merged

    def compaction_replay_message(self) -> dict[str, Any] | None:
        """Build one replay turn and remove Anthropic's native block from later metadata."""
        replay = self.provider_compaction
        extra = self.reasoning_extra
        if replay is None or not isinstance(extra, dict):
            return None if replay is None else {"role": "assistant", "content": [replay]}
        anthropic = extra.get("anthropic")
        native_content = anthropic.get("content") if isinstance(anthropic, dict) else None
        if not isinstance(native_content, list):
            return {"role": "assistant", "content": [replay]}
        native_compactions = [
            block
            for block in native_content
            if isinstance(block, dict) and block.get("type") == "compaction"
        ]
        if not native_compactions:
            return {"role": "assistant", "content": [replay]}
        self.reasoning_extra = {
            **extra,
            "anthropic": {
                **anthropic,
                "content": [
                    block
                    for block in native_content
                    if not (isinstance(block, dict) and block.get("type") == "compaction")
                ],
            },
        }
        return {
            "role": "assistant",
            "content": "",
            "extra_content": {
                "anthropic": {
                    **anthropic,
                    "content": [dict(native_compactions[-1])],
                }
            },
        }

    def note_hosted_tool_event(self, event: Any) -> None:
        """Record a provider-side tool call carried on ``_toolEvent``. These reach the client as
        their own frames but are not part of the assistant message this loop replays, so the
        follow-up request would lose whatever the provider just produced; Unsloth's own events
        carry a top-level ``type``, so ``_toolEvent`` is unambiguously the provider's. Both
        halves matter: ``tool_end`` generally omits ``tool_name``, and for Gemini code execution
        the code that ran is only in the ``tool_start`` arguments, so a result recorded alone is
        unlabelled."""
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "compaction_block":
            encrypted = event.get("encrypted_content")
            content = event.get("content")
            if isinstance(encrypted, str) and encrypted:
                self.provider_compaction = {
                    "type": "compaction",
                    "encrypted_content": encrypted,
                }
            elif isinstance(content, str) and content:
                self.provider_compaction = {"type": "compaction", "content": content}
            return
        call_id = event.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            return
        if kind not in ("tool_start", "tool_end"):
            return

        entry = self.hosted_results.setdefault(call_id, {})
        name = event.get("tool_name")
        if isinstance(name, str) and name:
            entry["name"] = name
        # OpenAI names an image generation's prompt only on the end event, so merge both halves.
        arguments = _hosted_arguments_for_model(event.get("arguments"))
        if arguments:
            merged = dict(entry.get("arguments_obj") or {})
            merged.update(arguments)
            entry["arguments_obj"] = merged
            # Truncate with a notice: Anthropic passes the whole tool input, and a silent cut misleads.
            entry["arguments"] = _truncate_for_model(
                json.dumps(merged, separators = (",", ":")),
                _HOSTED_ARGUMENT_MAX_CHARS,
                joiner = " ",
            )

        if kind == "tool_start":
            return
        result = event.get("result")
        if isinstance(result, str):
            # Gemini reports code that printed nothing as ""; track ended separately from result.
            entry["ended"] = True
        if isinstance(result, str) and result.strip():
            if _carries_image_sentinel(result):
                entry["produced_image"] = True
            # Strip data URIs as for local results; only sandbox tools emit __FILES__.
            stripped = strip_result_for_model(result, entry.get("name"))
            if stripped.strip():
                entry["result"] = _truncate_for_model(stripped)
        if event.get("image_b64"):
            entry["produced_image"] = True

    def hosted_replay_text(self) -> str:
        """The provider-run calls of this turn, as prose for the next request."""
        blocks: list[str] = []
        for entry in self.hosted_results.values():
            result = entry.get("result", "")
            produced_image = entry.get("produced_image")
            if not result and not produced_image and not entry.get("ended"):
                continue
            name = entry.get("name") or "tool"
            header = f"[{name} result]"
            arguments = entry.get("arguments")
            if arguments:
                header = f"[{name} {arguments}]"
            body = result or ("(produced an image)" if produced_image else "(no output)")
            if result and produced_image:
                body = f"{result}\n(produced an image)"
            blocks.append(f"{header}\n{body}")
        return "\n\n".join(blocks)

    def merge_structured(self, raw_calls: list[Any]) -> None:
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            index = raw_call.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                # Some servers stamp index only on the first fragment; continue the open call.
                index = self.last_index if self.last_index is not None else len(self.order)
            call_id = raw_call.get("id")
            # index restarts at 0 every tool round, so bare fragments belong to the newest call.
            key: Any = self.open_key_by_index.get(index, index)
            if isinstance(call_id, str) and call_id:
                owner = self.key_by_call_id.get(call_id)
                if owner is not None:
                    key = owner
                elif self.by_index.get(key, {}).get("id"):
                    # Two calls at one index: key the second on its id rather than merge their JSON.
                    key = (index, call_id)
            # A closed object takes no more content, so the next arguments belong to the next call.
            held = self.by_index.get(key)
            new_function = raw_call.get("function")
            new_arguments = _argument_fragment(
                new_function.get("arguments") if isinstance(new_function, dict) else None
            )
            new_name = new_function.get("name") if isinstance(new_function, dict) else None
            # Only a fragment starting with "{" opens a next call; an id naming a different call does too.
            held_name_now = held["function"]["name"] if held is not None else ""
            id_names_another_call = bool(
                isinstance(call_id, str)
                and call_id
                and isinstance(new_name, str)
                and new_name
                and held_name_now
                # Not a prefix test: "web" and "web_search" can both exist.
                and new_name != held_name_now
            )
            resends_this_call = bool(
                held is not None
                and isinstance(call_id, str)
                and call_id
                and not held.get("id")
                and isinstance(new_name, str)
                and new_name == held_name_now
                and isinstance(new_arguments, str)
                and new_arguments == held["function"]["arguments"]
            )
            names_next_call = bool(
                isinstance(new_name, str)
                and new_name
                and held_name_now
                # llama-server and vLLM resend the same name; forking would run the call twice.
                and new_name != held_name_now
            )
            opens_next_call = (
                bool(isinstance(new_arguments, str) and new_arguments.strip().startswith("{"))
                or id_names_another_call
                or names_next_call
            ) and not resends_this_call
            announces_over_announcement = (
                held is not None
                and (held.get("announced_only") is True or held.get("resend_suspect") is True)
                and not held["function"]["arguments"]
                and isinstance(new_name, str)
                and bool(new_name)
                and new_name != held_name_now
                and isinstance(new_arguments, str)
                and new_arguments.strip().startswith("{")
            )
            names_this_call = (
                held is not None
                and isinstance(call_id, str)
                and bool(call_id)
                and held.get("id") == call_id
            )
            slot_is_closed = False
            if held is not None and not names_this_call:
                closed, unfinished = self._scan(key, held["function"]["arguments"])
                slot_is_closed = bool(closed) and not unfinished
            extra = raw_call.get("extra_content")
            # Metadata on a repeated name belongs to whichever call the next object opens.
            extra_is_ambiguous = bool(
                slot_is_closed
                and isinstance(new_name, str)
                and new_name
                and new_name == held_name_now
                and isinstance(extra, dict)
                and extra
            )
            if (slot_is_closed and opens_next_call) or announces_over_announcement:
                if announces_over_announcement and held is not None:
                    held["superseded"] = True
                self.split_seq += 1
                waiting = self.pending_extra.pop(key, None)
                key = (index, "_split", self.split_seq)
                if waiting:
                    extra = {**waiting, **extra} if isinstance(extra, dict) else waiting
            self.last_index = index
            self.open_key_by_index[index] = key
            if key not in self.by_index:
                opening_name = new_name if isinstance(new_name, str) and new_name else held_name_now
                self.by_index[key] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": opening_name, "arguments": ""},
                    # A zero-parameter tool sends "" rather than omitting arguments.
                    "announced_only": new_arguments is None and bool(opening_name),
                    "from_fork": held is not None,
                    "resend_suspect": bool(
                        opening_name
                        and held_name_now
                        and opening_name != held_name_now
                        and opening_name.startswith(held_name_now)
                    ),
                }
                self.seq_by_key[key] = self._next_seq()
                # First-seen order: provider indexes can be negative or out of order.
                self.order.append(key)
            current = self.by_index[key]
            if new_arguments is not None:
                current["announced_only"] = False
            if isinstance(call_id, str) and call_id:
                current["id"] = call_id
                self.key_by_call_id.setdefault(call_id, key)
            extra_before = current.get("extra_content")
            if extra_is_ambiguous:
                self.pending_extra[key] = {
                    **self.pending_extra.get(key, {}),
                    **extra,
                }
            elif isinstance(extra, dict) and extra:
                # Gemini 3 thoughtSignature is per call; the native translator rejects a replay without it.
                current["extra_content"] = {**current.get("extra_content", {}), **extra}
            function = raw_call.get("function")
            if isinstance(function, dict):
                # llama-server resends the whole growing name, OpenAI streams fragments: a fragment starting
                # with the current name replaces it, anything else appends.
                fragment = function.get("name")
                name_before = current["function"]["name"]
                if isinstance(fragment, str) and fragment and not (slot_is_closed and name_before):
                    if fragment.startswith(name_before):
                        current["function"]["name"] = fragment
                    else:
                        current["function"]["name"] = name_before + fragment
                if isinstance(new_arguments, str) and not resends_this_call:
                    current["function"]["arguments"] += new_arguments
                    if not (isinstance(call_id, str) and call_id):
                        # Id-less streams have no ids to fork on, so split glued argument objects.
                        self._fork_glued_arguments(
                            index,
                            key,
                            current,
                            name_before,
                            fragment if isinstance(fragment, str) else "",
                            extra_before,
                            extra if isinstance(extra, dict) and extra else None,
                        )

    def _scan(self, key: Any, text: str) -> tuple[list[str], str]:
        """``_split_top_level_json_objects(text)``, resuming the scan for ``key``. The same answer
        as scanning from byte zero, at the cost of the bytes this call added rather than of the
        whole accumulation."""
        scan = self.scan_by_key.get(key)
        if scan is None:
            scan = _BoundaryScan()
            self.scan_by_key[key] = scan
        return scan.feed(text)

    def _fork_glued_arguments(
        self,
        index: int,
        key: Any,
        current: dict[str, Any],
        name_before: str,
        incoming_name: str,
        extra_before: dict[str, Any] | None,
        incoming_extra: dict[str, Any] | None,
    ) -> None:
        """Give every call after the first in one slot a call of its own."""
        complete, tail = self._scan(key, current["function"]["arguments"])
        segments = complete + ([tail] if tail else [])
        if len(segments) < 2:
            return
        self.scan_by_key.pop(key, None)
        # Per-call metadata stays with one call: two calls sharing a thoughtSignature is rejected.
        current["function"]["arguments"] = segments[0]
        born_name = incoming_name or name_before
        current["function"]["name"] = name_before or born_name
        # Metadata belongs to the last call: Gemini checks the signature against the call it rides.
        if incoming_extra is not None:
            if extra_before:
                current["extra_content"] = extra_before
            else:
                current.pop("extra_content", None)
        open_key: Any = key
        for segment in segments[1:]:
            self.split_seq += 1
            born_key = (index, "_split", self.split_seq)
            self.by_index[born_key] = {
                "id": "",
                "type": "function",
                "function": {"name": born_name, "arguments": segment},
            }
            self.seq_by_key[born_key] = self._next_seq()
            if tail and segment is segments[-1]:
                self.open_tail_keys.add(born_key)
            self.order.append(born_key)
            open_key = born_key
        if incoming_extra is not None:
            self.by_index[open_key]["extra_content"] = dict(incoming_extra)
        self.open_key_by_index[index] = open_key

    def _call_is_finished(self, key: Any) -> bool:
        """Whether a call forked off an unfinished object has since closed it. Only ``length`` and
        ``content_filter`` mark a turn truncated, so a stream that stops after ``{"a":1}{`` looks
        complete; running the tool a second time on that lone brace is worse than dropping a call
        the model never finished writing."""
        if key not in self.open_tail_keys:
            return True
        closed, unfinished = _split_top_level_json_objects(
            self.by_index[key]["function"]["arguments"]
        )
        return bool(closed) and not unfinished

    def _next_seq(self) -> int:
        self.seq_counter += 1
        return self.seq_counter

    def calls(
        self,
        taken: set[str] | None = None,
        cards: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Every call this turn produced, with ids unique across the whole run. ``taken`` carries
        the ids already used by earlier turns: a provider that restarts its numbering each turn,
        and the healer (which always mints call_0 first), would otherwise put two different
        results under one id in the conversation replayed upstream. ``cards`` carries the card
        ids already handed out, because the client keeps one list of cards for the whole
        response, so a second round has to keep counting rather than start again at
        ``tool_call_0`` and reopen the first round's cards."""
        seen: set[str] = taken if taken is not None else set()
        painted: set[str] = cards if cards is not None else set()
        out: list[dict[str, Any]] = []
        # Sort on the sequence number alone: comparing calls raises.
        numbered = [
            (
                self.seq_by_key.get(key, position),
                key[0] if isinstance(key, tuple) else key,
                self.by_index[key],
            )
            for position, key in enumerate(self.order)
            if self._call_is_finished(key)
        ]
        ordered = [
            (index, call) for _, index, call in sorted(numbered, key = lambda triple: triple[0])
        ]
        # Superseded announcements and false resends were never calls; their metadata (Gemini
        # signatures) goes to the call they were mistaken for.
        for key, waiting in self.pending_extra.items():
            held = self.by_index.get(key)
            if held is not None and waiting:
                held["extra_content"] = {**held.get("extra_content", {}), **waiting}
        self.pending_extra.clear()
        kept: list[tuple[Any, dict[str, Any]]] = []
        for index, call in ordered:
            never_ran = not call["function"]["arguments"]
            if never_ran and (
                call.get("superseded") is True
                or call.get("resend_suspect") is True
                or (call.get("announced_only") is True and call.get("from_fork") is True)
            ):
                extra = call.get("extra_content")
                if isinstance(extra, dict) and extra and kept:
                    previous = kept[-1][1]
                    previous["extra_content"] = {**previous.get("extra_content", {}), **extra}
                continue
            kept.append((index, call))
        ordered = kept
        # Reserve provider ids first so minted card ids never collide; only calls _normalized_call keeps.
        painted.update(
            call["id"]
            for _, call in ordered
            if isinstance(call.get("id"), str) and call["id"] and _normalized_call(call) is not None
        )
        for position, (index, call) in enumerate(ordered + [(None, call) for call in self.healed]):
            streamed_id = call.get("id")
            normalized = _normalized_call(call, fallback_id = f"call_{self.round}_{position}")
            if normalized is None:
                continue
            if not (isinstance(streamed_id, str) and streamed_id):
                card_id = _mint_streamed_card_id(painted, index)
                painted.add(card_id)
                normalized["card_id"] = card_id
            if normalized["id"] in seen:
                # Card events keep the streamed id the client painted with; the renamed id is what replays.
                normalized["stream_id"] = normalized["id"]
                # The renamed id is replayed too, so keep counting up until unique.
                renamed = f"{normalized['id']}_{self.round}_{position}"
                attempt = 0
                while renamed in seen:
                    attempt += 1
                    renamed = f"{normalized['id']}_{self.round}_{position}_{attempt}"
                normalized["id"] = renamed
            seen.add(normalized["id"])
            out.append(normalized)
        return out


def _rewrite_content(payload: dict[str, Any], choice: dict[str, Any], text: str) -> str:
    """Re-emit a chunk with its content replaced by what the healer released."""
    new_delta = {key: value for key, value in choice.get("delta", {}).items() if key != "content"}
    if text:
        new_delta["content"] = text
    new_choice = {key: value for key, value in choice.items() if key != "delta"}
    new_choice["delta"] = new_delta
    new_payload = {key: value for key, value in payload.items() if key != "choices"}
    new_payload["choices"] = [new_choice] + list(payload.get("choices", [])[1:])
    return _sse(new_payload)


def _split_turn_end(
    payload: dict[str, Any], choice: dict[str, Any], delta: dict[str, Any]
) -> tuple[str | None, str]:
    """Separate a turn-ending chunk into what can be sent now and the reason to hold. Only the
    finish_reason has to wait for the healer to resolve; the content on that same chunk does not,
    and holding it too would let residue flushed by ``finalize`` overtake it and reverse the text
    on the wire. So the content goes out in place and a bare finish-only chunk is what gets
    parked."""
    now_choice = {key: value for key, value in choice.items() if key != "finish_reason"}
    now_choice["delta"] = delta
    now_payload = {key: value for key, value in payload.items() if key != "choices"}
    now_payload["choices"] = [now_choice] + list(payload.get("choices", [])[1:])
    now = _sse(now_payload) if (delta or len(now_payload["choices"]) > 1) else None

    held_payload = {key: value for key, value in payload.items() if key != "choices"}
    held_payload["choices"] = [
        {
            "index": choice.get("index", 0),
            "delta": {},
            "finish_reason": choice.get("finish_reason"),
        }
    ]
    return now, _sse(held_payload)


def _unrun_provenance(tool_name: str, round_id: int) -> dict[str, Any]:
    """Provenance for a hand-built unrun card; carries the MCP display names so a budget-exhausted
    or truncated MCP call never shows the internal server id or an alias."""
    provenance: dict[str, Any] = {"source": "local", "round_id": round_id}
    mcp = mcp_display_parts(tool_name)
    if mcp:
        provenance["mcp_server"] = mcp[0]
        provenance["mcp_tool"] = mcp[1]
    return provenance


def _unrun_call_card(
    *, tool_name: str, tool_call_id: str, arguments: Any, result: str, provenance: dict[str, Any]
) -> list[str]:
    """The open/close pair for a call this loop announces but never runs. The close on its own is
    not enough: a call the provider streamed as a tool_calls delta already has a card, and the
    client reconciles both events onto it by id, but a call the healer promoted out of TEXT was
    never streamed as a delta, so a lone tool_end names a card that does not exist and the
    adapter drops it, leaving the user told nothing at all. Opening the card first makes both
    cases end the same way, and keeps the invariant the loop is tested on, that every tool_end
    closes a tool_start."""
    shown = arguments if isinstance(arguments, dict) else {}
    return [
        _sse(
            {
                "type": "tool_start",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": shown,
                "arguments_text": canonical_arguments_text(shown),
                "provenance": provenance,
            }
        ),
        _sse(
            {
                "type": "tool_end",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "result": result,
                "provenance": provenance,
            }
        ),
    ]


def _is_strict_prefix_of_declared(name: str, declared_names: set[str]) -> bool:
    return any(other != name and other.startswith(name) for other in declared_names)


def _mcp_provenance_by_id(
    turn: "_Turn", declared_names: set[str], stamped: set[str]
) -> dict[str, Any]:
    """MCP provenance per call id once its whole name has streamed, once per id.

    Declared catalog decides completeness (``mcp__srv__cre`` is well formed too); strict-prefix names wait for tool_start.
    """
    stamps: dict[str, Any] = {}
    for call in turn.by_index.values():
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in stamped:
            continue
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or name not in declared_names:
            continue
        if _is_strict_prefix_of_declared(name, declared_names):
            continue
        stamped.add(call_id)
        if not mcp_display_parts(name):
            continue
        stamps[call_id] = provisional_tool_provenance(name)
    return stamps


def _status_sse(text: str) -> str:
    """Tool badge text, in the shape the chat client already parses."""
    return _sse({"type": "tool_status", "content": text})


def _merge_usage(totals: dict[str, Any], usage: Any) -> None:
    """Sum one turn's usage into the running total. Per-turn usage is withheld while the loop runs
    and one summed chunk is sent at the end, so a multi-turn answer reports the same shape a
    single-turn one does instead of a burst of partial counts the client would have to add up.
    Detail sub-objects are summed too: reporting less than the same provider's plain stream would
    understate cost and cache hits."""
    if not isinstance(usage, dict):
        return
    for field, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            totals[field] = totals.get(field, 0) + value
            continue
        if field not in _USAGE_DETAIL_FIELDS or not isinstance(value, dict):
            continue
        bucket = totals.setdefault(field, {})
        if not isinstance(bucket, dict):
            continue
        for detail, count in value.items():
            if isinstance(count, int) and not isinstance(count, bool):
                bucket[detail] = bucket.get(detail, 0) + count


def _usage_chunk_line(
    model: str, totals: dict[str, Any], timings: dict[str, Any] | None
) -> str | None:
    if not totals:
        return None
    chunk: dict[str, Any] = {
        "id": "chatcmpl-external-tools",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [],
        "usage": totals,
    }
    if timings is not None:
        chunk["timings"] = timings
    return _sse(chunk)


def _is_usage_only(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    return "usage" in payload and isinstance(choices, list) and not choices


def _replayed_call_ids(conversation: list[dict[str, Any]]) -> set[str]:
    """Every tool-call id already in the history this run starts from. The healer restarts its
    counter every request, so a freshly minted call_0 collides with a stripped call_0 replayed
    from history inside one upstream body. Seeding the ledger makes calls() rename the new one as
    it does any repeat within a run."""
    taken: set[str] = set()
    for message in conversation:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]:
                taken.add(call["id"])
        result_id = message.get("tool_call_id")
        if isinstance(result_id, str) and result_id:
            taken.add(result_id)
    return taken


def _openai_compaction_item(event: Any) -> str | None:
    if not isinstance(event, dict) or event.get("type") != "compaction_block":
        return None
    encrypted = event.get("encrypted_content")
    return encrypted if isinstance(encrypted, str) and encrypted else None


def _with_openai_compaction(
    conversation: list[dict[str, Any]], compaction: tuple[list[dict[str, Any]], str] | None
) -> list[dict[str, Any]]:
    """Later rounds replay an OpenAI Responses compaction item after the messages it covers, or the provider compacts
    the same history again."""
    if compaction is None:
        return conversation
    sent, encrypted = compaction
    # A continuation merges into the last message sent rather than appending, and that message is no longer covered.
    covered = next(
        (index for index, message in enumerate(sent) if conversation[index] is not message),
        len(sent),
    )
    carrier = {
        "role": "assistant",
        "content": "",
        "extra_content": {"openai_responses_compaction": encrypted},
    }
    return [*conversation[:covered], carrier, *conversation[covered:]]


def _append_user_turn(conversation: list[dict[str, Any]], content: str) -> None:
    """Append a user turn, merging into a trailing one so roles keep alternating. A turn whose only
    calls were no-ops appends no assistant message, so a bare append would leave two user turns
    in a row. Unlike the in-process loops this conversation is rendered by the provider, and a
    strict server rejects that."""
    if not content:
        return
    last = conversation[-1] if conversation else None
    if (
        isinstance(last, dict)
        and last.get("role") == "user"
        and isinstance(last.get("content"), str)
    ):
        conversation[-1] = {**last, "content": f"{last['content']}\n\n{content}"}
        return
    conversation.append({"role": "user", "content": content})


def _advance_tool_stream(generator: Any, outcome: dict[str, Any]) -> Any:
    try:
        return next(generator)
    except StopIteration as stop:
        outcome["result"] = stop.value
        return _STEP_DONE


async def _drain_step_task(task: Any, cancel_event: threading.Event) -> None:
    """Join a pending ``next(gen)`` worker before its generator is closed. Cancelling the awaiting
    task does not stop the worker thread, and calling close() while next() is still running
    raises "generator already executing" and skips the generator's own cleanup. Setting the
    cancel flag lets a cancel-observing tool return, then the task is shielded until it finishes."""
    if task is None:
        return
    if task.done():
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass
        return
    cancel_event.set()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancel_event.set()
            continue
        except Exception:
            break
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


async def stream_with_studio_tools(
    transport: ToolLoopTransport,
    *,
    run: ToolLoopRun,
    policy: ToolLoopPolicy,
    cancel_event: threading.Event,
    mcp_image = None,
) -> AsyncIterator[str]:
    conversation = [dict(message) for message in run.messages]
    if mcp_image is not None:
        targets = await asyncio.to_thread(mcp_image_targets, sorted(_tool_names(policy.tools)))
        conversation = note_attached_image(conversation, targets)
    openai_compaction: tuple[list[dict[str, Any]], str] | None = None
    resumes_partial = run.continue_final_message
    # cap run-owned image parts across the full loop without counting caller attachments.
    loop_mcp_image_parts: list = list(run.promoted_image_parts)
    # preserve the request branch for tools that search conversation history.
    request_branch = list(run.messages)
    remaining = policy.max_calls
    unlimited = remaining >= 9999
    session_id = run.session_id
    thread_id = run.thread_id
    tools = policy.tools
    tool_choice = run.tool_choice if run.tool_choice is not None else "auto"
    allowed_tool_names = _tool_names(tools)
    tool_call_timeout = policy.timeout
    from state.tool_policy import (
        account_tool_stream,
        needs_tool_confirmation,
        normalize_sandbox_level,
        normalize_tool_permissions,
        requires_os_isolation,
        runs_without_os_sandbox,
    )

    permission_mode, bypass_permissions = normalize_tool_permissions(
        policy.permission_mode, policy.bypass_permissions
    )
    scoped_tool_stream = account_tool_stream(stream_tool_execution)
    confirm_tool_calls = policy.confirm_calls
    rag_scope = policy.rag_scope
    sandbox_level = normalize_sandbox_level(policy.sandbox_level)

    from core.inference.skill_mentions import load_mentioned_skills

    skill_loads = load_mentioned_skills(
        conversation,
        tools if tool_choice != "none" and (unlimited or remaining > 0) else [],
        permission_mode = permission_mode,
        bypass_permissions = bypass_permissions,
        confirm_tool_calls = confirm_tool_calls,
        session_id = session_id,
        cancel_event = cancel_event,
        continue_final_message = run.continue_final_message,
    )
    load_task = None
    flush_approval = False
    try:
        while True:
            load_task = asyncio.ensure_future(asyncio.to_thread(next, skill_loads, _STEP_DONE))
            timeout = _TOOL_APPROVAL_FLUSH_DELAY_S if flush_approval else TOOL_HEARTBEAT_INTERVAL_S
            done, _pending = await asyncio.wait({load_task}, timeout = timeout)
            while not done:
                yield _SSE_KEEPALIVE
                done, _pending = await asyncio.wait({load_task}, timeout = TOOL_HEARTBEAT_INTERVAL_S)
            event = load_task.result()
            load_task = None
            if event is _STEP_DONE:
                break
            flush_approval = event.get("status") == "awaiting_approval"
            yield _sse(event)
    finally:
        await _drain_step_task(load_task, cancel_event)
        skill_loads.close()
    # Never None: an unrestricted parse re-opens markerless tool-call promotion.
    heal_names = (
        heal_gate(policy.auto_heal, tools, tool_choice) if transport.heals_text_tool_calls else None
    )
    transport_sanitizes = bool(getattr(transport, "sanitizes_provider_frames", False))

    skip_autoinject = run.continue_final_message or (
        confirm_tool_calls and not bypass_permissions and permission_mode not in ("auto", "off")
    )
    autoinject = (
        None
        if skip_autoinject
        else await asyncio.to_thread(build_rag_autoinject, conversation, rag_scope)
    )
    if autoinject:
        for event in autoinject["events"]:
            yield _sse(event)
        conversation.extend(autoinject["messages"])

    round_id = 0
    executed_any = False
    model_name = run.model or "external"
    usage_totals: dict[str, Any] = {}
    last_timings: dict[str, Any] | None = None
    controller = ToolLoopController(
        tools = tools,
        auto_heal_tool_calls = policy.auto_heal is not False,
        deduplicate_tool_calls = policy.deduplicate_tool_calls is not False,
        session_id = session_id,
        thread_id = thread_id,
    )
    tool_hint = ", ".join(sorted(allowed_tool_names))
    reprompts = 0
    max_reprompts = MAX_ACT_REPROMPTS
    last_reprompt_text = ""
    provider_turns = 0
    used_call_ids: set[str] = _replayed_call_ids(conversation)
    # The client keeps one card list across rounds, so numbering carries over.
    painted_card_ids: set[str] = set()
    spent_budget_passes = 0
    fruitless_turns = 0
    # Headroom for no-op, nudge and final-answer passes; fruitless_turns ends idle runs.
    max_provider_turns = max(1, remaining) + 2 * MAX_ACT_REPROMPTS + 4

    while not cancel_event.is_set():
        if provider_turns >= max_provider_turns:
            break
        provider_turns += 1
        turn = _Turn(round = provider_turns)
        last_timings = None
        mcp_stamped_ids: set[str] = set()
        healer = StreamToolCallHealer(heal_names, tools) if heal_names else None
        # Hold the turn-ending chunk until finalize() says whether a healed call was promoted, then
        # arm the headerless stripper before releasing it.
        held_final: str | None = None

        active_tools = controller.active_tools()
        tools_available = (
            tool_choice != "none" and bool(active_tools) and (unlimited or remaining > 0)
        )
        # Withdraw the catalog and pin "none" so the model cannot call a tool it was just denied.
        turn_tool_choice = tool_choice if tools_available else "none"
        if executed_any and turn_tool_choice not in ("auto", "none"):
            turn_tool_choice = "auto"

        sent_messages = list(conversation)
        generator = transport.stream(
            messages = _with_openai_compaction(conversation, openai_compaction),
            tools = active_tools if tools_available else None,
            tool_choice = turn_tool_choice,
            cancel_event = cancel_event,
        )
        try:
            async for line in generator:
                if _is_done_sentinel(line):
                    continue
                # Strip provider copies of our card vocabulary, or they paint cards for tools that never ran.
                # Skipped when the transport already did it: its own frames would be dropped.
                if not transport_sanitizes:
                    sanitized = sanitize_provider_sse_line(line)
                    if sanitized is None:
                        continue
                    line = sanitized
                payload = _chunk_payload(line)
                if payload is None:
                    yield line
                    continue
                # Routers name the concrete model on each chunk; prefer it for the summed usage chunk.
                upstream_model = payload.get("model")
                if isinstance(upstream_model, str) and upstream_model:
                    model_name = upstream_model
                if "usage" in payload:
                    _merge_usage(usage_totals, payload.get("usage"))
                    if isinstance(payload.get("timings"), dict):
                        last_timings = payload["timings"]
                    if _is_usage_only(payload):
                        continue
                    payload.pop("usage", None)
                    payload.pop("timings", None)
                    line = "data: " + json.dumps(payload, separators = (",", ":"))
                choices = payload.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else {}
                if not isinstance(choice, dict):
                    yield line
                    continue

                delta = choice.get("delta")
                delta = delta if isinstance(delta, dict) else {}
                reasoning = delta.get("reasoning_content")
                if getattr(transport, "preserves_reasoning", False) and isinstance(reasoning, str):
                    turn.reasoning.append(reasoning)
                content = delta.get("content")
                raw_calls = delta.get("tool_calls")
                extra = delta.get("extra_content")
                turn.note_reasoning_extra(extra)
                turn.note_hosted_tool_event(payload.get("_toolEvent"))
                compaction = _openai_compaction_item(payload.get("_toolEvent"))
                if compaction:
                    openai_compaction = (sent_messages, compaction)
                if isinstance(choice.get("finish_reason"), str):
                    turn.finish_reason = choice["finish_reason"]
                hold_final = (
                    isinstance(choice.get("finish_reason"), str)
                    and healer is not None
                    and not healer.dormant
                )

                if isinstance(raw_calls, list) and raw_calls:
                    if healer is not None and not healer.dormant:
                        # Structured calls work, so the held text was prose; release it to the client.
                        for kind, value in healer.structured_tool_call_seen():
                            if kind == "text" and value:
                                turn.text.append(value)
                                yield _sse({"choices": [{"index": 0, "delta": {"content": value}}]})
                    turn.merge_structured(raw_calls)
                    stamps = _mcp_provenance_by_id(turn, allowed_tool_names, mcp_stamped_ids)
                    if stamps:
                        payload["_mcp_provenance"] = stamps
                        line = "data: " + json.dumps(payload, separators = (",", ":"))

                if healer is None or healer.dormant or not isinstance(content, str) or not content:
                    plain = _delta_text(content)
                    if plain:
                        turn.text.append(plain)
                    if hold_final:
                        now, held_final = _split_turn_end(payload, choice, delta)
                        if now is not None:
                            yield now
                    else:
                        yield line
                    continue

                released: list[str] = []
                for kind, value in healer.feed(content):
                    if kind == "text":
                        if value:
                            released.append(value)
                    elif kind == "tool_call":
                        turn.healed.append(value)
                visible = "".join(released)
                if visible:
                    turn.text.append(visible)
                if visible == content:
                    if hold_final:
                        now, held_final = _split_turn_end(payload, choice, delta)
                        if now is not None:
                            yield now
                    else:
                        yield line
                    continue
                if visible or turn.finish_reason is not None or len(delta) > 1:
                    if hold_final:
                        healed_delta = {
                            key: value for key, value in delta.items() if key != "content"
                        }
                        if visible:
                            healed_delta["content"] = visible
                        now, held_final = _split_turn_end(payload, choice, healed_delta)
                        if now is not None:
                            yield now
                    else:
                        yield _rewrite_content(payload, choice, visible)

            if healer is not None:
                for kind, value in healer.finalize():
                    if kind == "text":
                        if value:
                            turn.text.append(value)
                            yield _sse({"choices": [{"index": 0, "delta": {"content": value}}]})
                    elif kind == "tool_call":
                        turn.healed.append(value)

            # Arm even without held_final: a provider closing on [DONE] alone would otherwise end with no
            # finish_reason, which openai-node rejects. Truncated turns keep their reason.
            if (
                turn.healed
                and turn.finish_reason not in ("length", "content_filter")
                and policy.on_withheld_tool_call is not None
            ):
                policy.on_withheld_tool_call()
            if held_final is not None:
                yield held_final
                held_final = None

        finally:
            # Release the upstream now; async-generator finalisation runs after the route closed.
            aclose = getattr(generator, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except (RuntimeError, GeneratorExit):
                    pass

        # Explicit: a turn closed on [DONE] alone carries no finish_reason.
        if policy.on_provider_turn_end is not None:
            policy.on_provider_turn_end()

        if turn.provider_compaction is not None:
            compaction_message = turn.compaction_replay_message()
            assert compaction_message is not None
            system_messages = [
                message
                for message in conversation
                if isinstance(message, dict) and message.get("role") == "system"
            ]
            conversation[:] = [
                *system_messages,
                compaction_message,
            ]
            # The partial this run resumed is inside the compaction now, and merging over the item would discard it.
            resumes_partial = False

        # Both of these mean the turn ended before the model finished saying what it wanted: "length" hit the token
        # ceiling, "content_filter" had the output cut by the provider's own filter. Either way a call collected so
        # far may be half-written, so it is described rather than run. "stop" is not in this set: llama.cpp and vLLM
        # routinely finish a perfectly good tool call with it, and refusing those would disable tool calling on
        # exactly the self-hosted servers this path exists for.
        truncated = turn.finish_reason in ("length", "content_filter")
        if truncated and healer is not None and turn.healed:
            # Truncated calls must not run; give back the exact span the healer removed so the text survives.
            for healed_call in turn.healed:
                span = healer.promoted_source(healed_call.get("id", ""))
                if not span:
                    continue
                turn.text.append(span)
                yield _sse({"choices": [{"index": 0, "delta": {"content": span}}]})
        # Truncation wins: half-written arguments are never shown.
        unrun_reason = None
        if truncated:
            unrun_reason = _TOOL_TRUNCATED
        elif tool_choice == "none":
            unrun_reason = _TOOL_CHOICE_NONE
        if unrun_reason is not None:
            # The relayed delta already drew a card: close it like any unrun call, via `calls` (executes nothing), which
            # mints the id the client drew for an id-less call.
            for raw_call in turn.calls(used_call_ids, painted_card_ids):
                unrun_id = raw_call.get("card_id") or raw_call.get("stream_id") or raw_call["id"]
                name = raw_call["function"]["name"]
                for card_line in _unrun_call_card(
                    tool_name = name,
                    tool_call_id = unrun_id,
                    arguments = {} if truncated else raw_call.get("arguments"),
                    result = unrun_reason,
                    provenance = _unrun_provenance(name, round_id + 1),
                ):
                    yield card_line
        # tool_choice "none" is an instruction, and a provider that emits a call anyway has not been authorized to run
        # one. Withdrawing the catalog on the way out is not enough on its own: Deep Research sets "none" exactly so
        # the scraped web text in its prompts cannot reach python or terminal, so a naive or compromised endpoint
        # echoing a call back must not be able to execute it here.
        calls = [] if unrun_reason is not None else turn.calls(used_call_ids, painted_card_ids)
        if not calls:
            # Clear the badge so the client closes a refused call's card; [DONE]-only streams have no other
            # boundary.
            if turn.by_index:
                yield _status_sse("")
            # A model that only announced a tool gets one nudge, as in the local loops.
            visible_answer = "".join(turn.text)
            if (
                tools_available
                and nudge_enabled(policy.nudge_tool_calls)
                and not controller.force_final_answer
                and reprompts < max_reprompts
                and is_short_intent_without_action(visible_answer)
                and not is_reprompt_repeat(visible_answer, last_reprompt_text)
            ):
                reprompts += 1
                last_reprompt_text = visible_answer
                stalled_hosted = turn.hosted_replay_text()
                if stalled_hosted:
                    stalled_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": (
                            f"{visible_answer}\n\n{stalled_hosted}"
                            if visible_answer
                            else stalled_hosted
                        ),
                    }
                    if turn.reasoning_extra:
                        # Gemini 3 needs the text part's thoughtSignature on replay.
                        stalled_message["extra_content"] = turn.reasoning_extra
                    append_assistant_turn(
                        conversation,
                        stalled_message,
                        # A resumed partial is the same turn as what the model just added, so merge rather than
                        # append: appending puts a turn boundary mid-sentence.
                        continue_final_message = resumes_partial,
                    )
                _append_user_turn(conversation, reprompt_to_act_message(tool_hint))
                continue
            break

        round_id += 1
        assistant_tool_calls: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        noop_messages: list[dict[str, Any]] = []
        # Per result, so the per-result image quota is not spent on the first call of a batch.
        turn_mcp_images: list[list[dict[str, Any]]] = []
        turn_executed_real_tool = False

        for call in calls:
            if cancel_event.is_set():
                break
            # Before the gate: exhausted calls are replayed too, and only the decision's replay parses.
            decision = controller.prepare_call(call)
            signed_provider_call = _signed_provider_call_for_replay(call)
            if not unlimited and remaining <= 0:
                for card_line in _unrun_call_card(
                    tool_name = call["function"]["name"],
                    tool_call_id = call.get("card_id") or call.get("stream_id") or call["id"],
                    arguments = decision.tool_start_payload()["arguments"],
                    result = _TOOL_BUDGET_EXHAUSTED,
                    provenance = _unrun_provenance(call["function"]["name"], round_id),
                ):
                    yield card_line
                # The result below has to be replayed with its call: only the call that spent the last slot reaches
                # assistant_tool_calls further down, so this one would arrive as an orphan role="tool" message and
                # OpenAI, Anthropic and Gemini all reject that history instead of answering.
                exhausted_call = signed_provider_call or decision.as_assistant_tool_call()
                exhausted_extra = call.get("extra_content")
                if (
                    signed_provider_call is None
                    and isinstance(exhausted_extra, dict)
                    and exhausted_extra
                ):
                    exhausted_call["extra_content"] = exhausted_extra
                assistant_tool_calls.append(exhausted_call)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["function"]["name"],
                        "content": _TOOL_BUDGET_EXHAUSTED,
                    }
                )
                continue
            # The frontend groups a round by this id (codexLocalToolRoundId).
            decision.provenance["round_id"] = round_id
            image_share = None
            if decision.should_execute and mcp_image is not None:
                image_share = await asyncio.to_thread(
                    mcp_image_share, decision.tool_name, decision.arguments, mcp_image
                )
                if image_share is not None:
                    decision = controller.reprepare_call(decision)
            if not decision.should_execute:
                completion = controller.record_noop(decision)
                if getattr(transport, "tool_result_only_continuation", False):
                    assistant_tool_calls.append(
                        signed_provider_call or decision.as_assistant_tool_call()
                    )
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": decision.tool_call_id,
                            "content": completion.model_message()["content"],
                        }
                    )
                else:
                    noop_messages.append(completion.model_message())
                # Close the provider-streamed card for enabled tools; an unknown tool gets no card.
                if decision.action == "disabled":
                    continue
                for card_line in _unrun_call_card(
                    tool_name = decision.tool_name,
                    tool_call_id = (
                        call.get("card_id") or call.get("stream_id") or decision.tool_call_id
                    ),
                    arguments = decision.arguments,
                    result = _TOOL_SKIPPED.get(decision.action, "Unsloth did not run this call."),
                    provenance = decision.provenance,
                ):
                    yield card_line
                continue
            assistant_call = signed_provider_call or decision.as_assistant_tool_call()
            call_extra = call.get("extra_content")
            if signed_provider_call is None and isinstance(call_extra, dict) and call_extra:
                assistant_call["extra_content"] = call_extra
            assistant_tool_calls.append(assistant_call)

            name = decision.tool_name
            arguments = decision.arguments
            call_id = decision.tool_call_id
            card_id = decision.card_id
            needs_confirmation = needs_tool_confirmation(
                confirm_tool_calls = confirm_tool_calls,
                bypass_permissions = bypass_permissions,
                permission_mode = permission_mode,
                name = name,
                arguments = arguments,
                is_high_risk = is_high_risk_tool_call,
                never_needs = never_needs_approval,
                sandbox_level = sandbox_level,
            )
            # Sending the user's image always asks, whatever the permission mode.
            needs_confirmation = needs_confirmation or image_share is not None
            strict_isolation = requires_os_isolation(
                confirm_tool_calls = confirm_tool_calls,
                bypass_permissions = bypass_permissions,
                permission_mode = permission_mode,
                name = name,
                arguments = arguments,
                prompted = needs_confirmation,
                is_high_risk = is_high_risk_tool_call,
                sandbox_level = sandbox_level,
            )
            approval_id = new_approval_id() if needs_confirmation else ""
            decision_slot = (
                begin_tool_decision(session_id, approval_id) if needs_confirmation else None
            )

            start_event = decision.tool_start_event()
            start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirmation
            if image_share is not None:
                start_event["image_disclosure"] = image_share["disclosure"]
            denied = False
            try:
                yield _status_sse(
                    awaiting_approval_status(name) if needs_confirmation else decision.status_text
                )
                yield _sse(start_event)
                verdict = None
                denied_reason = None
                if decision_slot is not None:
                    waiter = asyncio.ensure_future(
                        asyncio.to_thread(
                            wait_tool_decision, decision_slot, approval_id, cancel_event
                        )
                    )
                    try:
                        done, _pending = await asyncio.wait(
                            {waiter}, timeout = _TOOL_APPROVAL_FLUSH_DELAY_S
                        )
                        while not done:
                            yield _SSE_KEEPALIVE
                            done, _pending = await asyncio.wait(
                                {waiter}, timeout = TOOL_HEARTBEAT_INTERVAL_S
                            )
                    finally:
                        if not waiter.done():
                            waiter.cancel()
                    verdict = waiter.result() if waiter.done() else None
                if verdict == "deny":
                    denied_reason = decision_reason(decision_slot)
                    decision_slot = None
                    denied = True
                elif verdict is not None:
                    yield _status_sse(decision.status_text)
                if not denied:
                    decision_slot = None
            finally:
                if decision_slot is not None:
                    abort_tool_decision(decision_slot, approval_id)

            if denied:
                # An unanswered approval is not a user decision; do not say the user declined.
                denied_text = (
                    TOOL_APPROVAL_EXPIRED_MESSAGE
                    if denied_reason == DECISION_EXPIRED
                    else TOOL_REJECTED_MESSAGE
                )
                yield _sse(
                    {
                        "type": "tool_end",
                        "tool_name": name,
                        "tool_call_id": card_id,
                        "result": denied_text,
                        "provenance": decision.provenance,
                    }
                )
                denied_message: dict[str, Any] = {
                    "role": "tool",
                    "name": name,
                    "content": denied_text,
                }
                if call_id:
                    denied_message["tool_call_id"] = call_id
                tool_messages.append(denied_message)
                reprompts = max_reprompts
                continue

            # Only a call the user answered: the executor lets it reach the host paths it names.
            host_access_approved = verdict not in (None, "deny")

            def _invoke(
                output_callback: Any,
                call = decision,
                approved = host_access_approved,
                strict = strict_isolation,
            ) -> str:
                kwargs: dict[str, Any] = {
                    "cancel_event": cancel_event,
                    "timeout": None if tool_call_timeout >= 9999 else tool_call_timeout,
                    "session_id": session_id,
                    "thread_id": thread_id,
                    "rag_scope": rag_scope,
                    "disable_sandbox": bypass_permissions,
                }
                # Run unasked only because the OS sandbox was on: refuse if it is not any more.
                if strict and accepts_kwarg(execute_tool, "tool_execution_mode"):
                    kwargs["tool_execution_mode"] = "required"
                elif runs_without_os_sandbox(call.tool_name, sandbox_level) and accepts_kwarg(
                    execute_tool, "tool_execution_mode"
                ):
                    kwargs["tool_execution_mode"] = "software"
                # Provider loops share the local catalogue selector, so search_conversation is advertised here too
                # once a thread has an archive and needs the same branch: the stored rows are the whole DAG, and Retry
                # leaves the replaced response in them.
                if accepts_kwarg(execute_tool, "conversation_branch"):
                    kwargs["conversation_branch"] = request_branch
                if approved and accepts_kwarg(execute_tool, "host_access_approved"):
                    kwargs["host_access_approved"] = True
                # External window is unknowable; 0 keeps the default page cap.
                if accepts_kwarg(execute_tool, "context_tokens"):
                    kwargs["context_tokens"] = 0
                if accepts_kwarg(execute_tool, "conversation_budget_tokens"):
                    try:
                        from core.rag import config as rag_config
                        kwargs["conversation_budget_tokens"] = max(
                            1, int(rag_config.CHUNK_TOKENS)
                        ) * max(1, int(rag_config.CONVERSATION_ARCHIVE_TOP_K))
                    except Exception:
                        pass
                if accepts_output_callback(execute_tool):
                    kwargs["output_callback"] = output_callback
                kwargs.update(search_images_kwargs(execute_tool, call.tool_name))
                if image_share is not None:
                    kwargs["mcp_image"] = image_share["image"]
                return execute_tool(call.tool_name, call.arguments, **kwargs)

            tool_stream = scoped_tool_stream(
                _invoke,
                tool_name = name,
                tool_call_id = card_id,
                cancel_event = cancel_event,
            )
            outcome: dict[str, Any] = {}
            step_task: Any = None
            try:
                while True:
                    if cancel_event.is_set():
                        # A tool ignoring the cancel event would heartbeat forever; let the bounded drain join it.
                        break
                    step_task = asyncio.create_task(
                        asyncio.to_thread(_advance_tool_stream, tool_stream, outcome)
                    )
                    # wait, not await: cancelling must leave the worker pending for the drain.
                    await asyncio.wait({step_task})
                    event = step_task.result()
                    step_task = None
                    if event is _STEP_DONE:
                        break
                    if isinstance(event, dict) and event.get("type") == "heartbeat":
                        yield _SSE_KEEPALIVE
                    else:
                        yield _sse(event)
                if "result" in outcome:
                    result = outcome["result"]
                elif cancel_event.is_set():
                    # Not "": that would record a successful empty result for an abandoned tool.
                    result = _TOOL_CANCELLED
                else:
                    result = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported back to the model
                result = f"Error: tool raised an exception: {exc}"
            finally:
                await _drain_step_task(step_task, cancel_event)
                tool_stream.close()

            completion = controller.record_result(decision, result)
            # Counted even on failure (side effects happened), per call so parallel calls each spend one.
            if not unlimited:
                remaining -= 1
            turn_executed_real_tool = True
            executed_any = True
            last_reprompt_text = ""
            yield _sse(completion.tool_end_event())
            tool_messages.append(completion.tool_message())
            # Off the loop: the envelope is a 12 MB json-load.
            _completion_images = (
                await asyncio.to_thread(completion.mcp_images) if run.supports_vision else []
            )
            if _completion_images:
                turn_mcp_images.append(_completion_images)

        yield _status_sse("")

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": strip_tool_markup(
                "".join(turn.text), final = True, enabled_tool_names = allowed_tool_names
            ),
        }
        hosted_text = turn.hosted_replay_text()
        if hosted_text:
            # Replay provider-hosted tool output as text: native shapes differ per provider.
            assistant_message["content"] = (
                f"{assistant_message['content']}\n\n{hosted_text}"
                if assistant_message["content"]
                else hosted_text
            )
        if turn.reasoning:
            assistant_message["reasoning_content"] = "".join(turn.reasoning)
        if turn.reasoning_extra:
            assistant_message["extra_content"] = turn.reasoning_extra
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls
        if assistant_message["content"] or assistant_tool_calls:
            append_assistant_turn(
                conversation,
                assistant_message,
                continue_final_message = resumes_partial,
            )
        conversation.extend(tool_messages)
        # After the results so a no-op never splits a call from them; merged to keep roles alternating.
        _append_user_turn(
            conversation,
            "\n\n".join(dict.fromkeys(message["content"] for message in noop_messages)),
        )
        if turn_mcp_images and run.supports_vision:
            # With several results "the tool call above" is ambiguous, so the block says so.
            from core.inference.mcp_images import DETACHED_IMAGE_TURN_TEXT
            _lead = DETACHED_IMAGE_TURN_TEXT if len(tool_messages) != 1 else None
            await asyncio.to_thread(
                _append_mcp_images_owned,
                conversation,
                turn_mcp_images,
                loop_mcp_image_parts,
                _lead,
            )

        if turn_executed_real_tool:
            max_reprompts = _MAX_POST_TOOL_REPROMPTS
            reprompts = 0
            fruitless_turns = 0
        else:
            fruitless_turns += 1
            if fruitless_turns >= _MAX_FRUITLESS_TURNS:
                break
        if remaining <= 0 and not unlimited and not controller.force_final_answer:
            if spent_budget_passes:
                break
            spent_budget_passes += 1
            if not getattr(transport, "tool_result_only_continuation", False):
                _append_user_turn(conversation, _BUDGET_EXHAUSTED_NUDGE)

    usage_line = _usage_chunk_line(model_name, usage_totals, last_timings)
    if usage_line is not None:
        yield usage_line
