# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth-owned tool execution loop, shared by every external provider transport.

The loop owns the parts that do not depend on how bytes reach the provider:
turn cycling, the tool budget, the approval handshake, local execution through
``execute_tool``, and the conversation replay that carries results back. A
``ToolLoopTransport`` supplies the one thing that does differ -- an async
iterator of OpenAI-shaped SSE lines for one turn.

Two transports exist: ``CodexTransport`` (the ChatGPT subscription, which speaks
its own app-server protocol) and ``OAICompatTransport`` (everything routed
through ``ExternalProviderClient``, which already normalises Anthropic, Gemini
and the Responses API into OpenAI chunk shape).

Self-hosted models frequently write tool calls as text rather than emitting
structured ``delta.tool_calls``. A transport that sets ``heals_text_tool_calls``
gets the content stream routed through ``StreamToolCallHealer``, the same
bounded buffer the client-tool passthrough uses: only a trailing partial-signal
window or a suspected tool block is ever withheld, and a block that cannot
become a declared call flushes verbatim. Nothing here invents a cap on how much
model output may be held.

Origins. This file is assembled out of four contributor PRs rather than written
from scratch, and the squash merge of #8665 did not carry their authorship, so
it is recorded here:

* #8626 (khalidejaz) -- the registry capability and the hidden-entry exposure,
  including the ``provider_runs_local_tools()`` name ``providers.py`` still
  uses.
* #8630 (mahiatlinux) -- most of the loop internals below: the user-turn append,
  the tool-stream advance and drain, usage merging, the approval flush delay,
  the budget-exhausted nudge, and the streamed tool-name rule that llama-server
  forced. Also the browser testing against a control worktree that caught a
  truncated turn discarding a healed call along with the text describing it.
* #7805 (Etherll) -- the hosted-provider reach.
* #7330 (Souravrajvi0) -- the original OpenAI-compatible framing.

Two ideas from those PRs were deliberately not taken, which is worth knowing
before anyone re-derives them: the per-connection opt-in from #8630 and the
``studio_tool_execution`` column from #7805. #8665 shipped a static disclosure
instead, while widening the capability from four self-hosted types to thirteen.
"""

from __future__ import annotations

import asyncio
import json
import threading

from dataclasses import dataclass, field
from collections.abc import AsyncIterator
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
from core.inference.tool_loop_controller import (
    ToolLoopController,
    awaiting_approval_status,
    mcp_display_parts,
    strip_result_for_model,
)
from core.inference.tool_stream_exec import (
    TOOL_HEARTBEAT_INTERVAL_S,
    accepts_kwarg,
    accepts_output_callback,
    search_images_kwargs,
    stream_tool_execution,
)
from core.inference.tools import build_rag_autoinject, execute_tool, is_high_risk_tool_call
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
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

# Card text for a call the controller skipped. The client already painted a card
# from the provider's own tool_calls delta, so it needs a short result; the long
# model-facing nudge stays in the conversation.
_TOOL_SKIPPED = {
    "duplicate": "Unsloth did not run this call because an identical one had already completed.",
    "disabled": _TOOL_DISABLED,
    "render_html_repeat": "Unsloth did not run this call because render_html already ran.",
}

# Verbatim from the local loops: the last pass answers instead of asking for more.
_BUDGET_EXHAUSTED_NUDGE = (
    "You have used all available tool calls. Based on everything you have found "
    "so far, provide your final answer now. Do not call any more tools."
)

# SSE comment written while a tool blocks, so a proxy cannot idle the stream out.
_SSE_KEEPALIVE = ": keep-alive"

# Delay before the first approval keepalive so it lands as a separate write and
# cannot coalesce with the gated card. Without the gap a desktop webview can hold
# both in one frame and the Allow / Deny buttons appear only after the tool ends.
_TOOL_APPROVAL_FLUSH_DELAY_S = 0.05

# A stall after a tool ran costs a re-run on retry, so allow one, as locally.
_MAX_POST_TOOL_REPROMPTS = 1

# Usage sub-objects worth summing rather than overwriting: the frontend reads the
# cache slice and pricing reads the reasoning slice.
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
    """Hold a hosted result to the same cap a local result gets.

    Read off ``tools`` rather than copied, so an install that lowers
    ``UNSLOTH_TOOL_RESULT_MAX_CHARS`` gets the lower cap here too.
    """
    if limit is None:
        limit = tools_module._MAX_OUTPUT_CHARS
    if len(text) <= limit:
        return text
    return text[:limit] + f"{joiner}... [truncated, {len(text) - limit} more characters]"


# Cap on a call's label. Small next to the result cap: this is the query or the
# code, not the output.
_HOSTED_ARGUMENT_MAX_CHARS = 2000


# Provider plumbing hung off ``arguments`` for the frontend and native-history
# replay, never for the model: Gemini's ``executableCode`` part plus an opaque
# ``thoughtSignature``, OpenAI's paired reasoning item with its multi-kilobyte
# ``encrypted_content``. As prose they are base64 cut off mid-token.
_HOSTED_ARGUMENT_PLUMBING_KEYS = frozenset({"google", "_server_tool"})


def _carries_image_sentinel(result: str) -> bool:
    """Whether a hosted result ends in the ``__IMAGES__`` envelope itself.

    Validated rather than matched on sight, the way the local strippers
    validate theirs: a fetched page that merely writes the marker is prose, and
    reading it as a picture would report an image the turn never made.
    """
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


# Consecutive turns that asked for a tool but ran none before the loop gives up.
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


def _normalized_call(call: dict[str, Any], fallback_id: str = "") -> dict[str, Any] | None:
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    if not isinstance(call_id, str) or not call_id:
        # The id is Unsloth's correlation key, not the model's contract. Several
        # OpenAI-compatible servers omit it; dropping the call lost a real
        # request with no error, so mint one instead.
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
        # Deeply nested but syntactically valid JSON blows the interpreter's
        # stack rather than failing to decode, and RecursionError is not a
        # ValueError. Uncaught it escapes the loop after the provider's delta
        # was already relayed, so the client gets a server error mid-stream
        # instead of a call that was simply refused.
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


def _delta_text(content: Any) -> str:
    """Text of a content delta, whether it is a plain string or content parts.

    Structured content blocks reach the client fine either way, but only the
    text of them belongs in the assistant message replayed upstream, and
    dropping it there loses the turn's prose on the follow-up call.
    """
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

    # Whether the provider may write tool calls as text instead of emitting
    # structured delta.tool_calls. Codex never does; a self-hosted GGUF often does.
    heals_text_tool_calls: bool

    # Whether the transport already stripped Unsloth's control vocabulary from
    # every raw upstream line. A transport that has not is sanitized here; one
    # that has must not be sanitized twice, because by this point its own
    # synthesized frames (a provider-hosted image result, say) are indis-
    # tinguishable from a forged one and would be thrown away.
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


@dataclass(frozen = True)
class ToolLoopPolicy:
    tools: list[dict[str, Any]]
    max_calls: int
    timeout: int
    permission_mode: str
    confirm_calls: bool
    bypass_permissions: bool
    rag_scope: dict[str, Any] | None
    # None means "follow the process default"; False disables text-form healing.
    auto_heal: bool | None = None
    # None follows UNSLOTH_TOOL_CALL_NUDGE; explicit booleans win.
    nudge_tool_calls: bool | None = None


def _reject_json_constant(name: str) -> Any:
    """Refuse ``NaN`` / ``Infinity``, which ``json.loads`` takes and JSON does not.

    ``JSON.parse`` has no such literals, so leaving them accepted here lets the
    two scanners cut the same text in different places: the backend would run
    two calls where the frontend shows one.
    """
    raise ValueError(f"{name} is not JSON")


def _split_top_level_json_objects(text: str) -> tuple[list[str], str]:
    """The top-level JSON objects in ``text``, and any object still unfinished.

    A call's ``function.arguments`` is one JSON object, so a second top-level
    ``{`` means one index slot took a second parallel call. Text that is not a
    run of whole objects (top-level array or scalar, trailing junk, unbalanced
    brace) comes back whole as the tail, so a stream this was never meant for
    is left alone.

    Mirrors ``splitTopLevelJsonObjects`` in
    ``studio/frontend/src/features/chat/tool-call-arguments.ts``: the two see
    the same deltas and have to agree on where a call ends.
    """
    unsplit: tuple[list[str], str] = ([], text)
    complete: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            # A backslash escapes one character, so a run of them toggles.
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if depth == 0:
            # Between objects only whitespace, "\r\n" as readily as "\n".
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
                        # Validation only, so the numbers are never read: keeping
                        # them as text sidesteps the 4300-digit cap on int
                        # conversion, which JSON.parse does not have and which
                        # would otherwise make the two disagree on where a call
                        # ends over a literal that fits in an ordinary payload.
                        parse_int = str,
                    )
                except (ValueError, TypeError):
                    # Balanced but invalid, so the brace count was a
                    # coincidence and cutting here would invent a call.
                    return unsplit
                except RecursionError:
                    # Deeply nested but valid JSON blows the interpreter's stack
                    # instead of failing to decode, and RecursionError is not a
                    # ValueError. Uncaught it escapes the loop mid-stream, so
                    # the segment counts as unsplittable and _normalized_call
                    # downgrades it to _raw as it always has.
                    return unsplit
                complete.append(segment)
                start = -1

    return complete, ("" if start == -1 else text[start:])


@dataclass
class _BoundaryScan:
    """``_split_top_level_json_objects`` over a string that only ever grows.

    One call's arguments arrive as many small fragments, and rescanning the
    whole accumulation per fragment makes streaming one argument of length N
    cost O(N^2): a 10 KB argument delivered a character at a time took about
    five seconds, which stalls the response. The scan resumes where it stopped
    instead, so the same argument costs one pass in total.

    ``feed`` must be given the same string extended, never a rewritten one --
    a fork rewrites a slot's arguments, so ``_Turn`` drops the scan for the
    keys it touches and lets the next fragment start a fresh one.
    """

    depth: int = 0
    start: int = -1
    in_string: bool = False
    escaped: bool = False
    scanned: int = 0
    complete: list[str] = field(default_factory = list)
    # Junk at depth 0 or a segment that does not parse makes the whole string
    # unsplittable, and appending to it can never make it splittable again.
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
    # call key each delta index maps to: the index itself until a second call
    # forks off it, then (index, call_id), or (index, "_split", n) for a call
    # that had no id to fork on and was found on a JSON object boundary.
    open_key_by_index: dict[int, Any] = field(default_factory = dict)
    last_index: int | None = None
    split_seq: int = 0
    # A name that arrived at a closed slot, waiting for the arguments that say
    # which call it names, and the metadata that arrived with it. Gemini stows
    # the thoughtSignature for the call being announced, so it has to travel
    # with the name rather than land on the call that has already closed.
    pending_name_by_index: dict[int, str] = field(default_factory = dict)
    pending_extra_by_index: dict[int, dict[str, Any]] = field(default_factory = dict)
    # When the announcement landed, so a call the stream announced before
    # another one still runs before it whenever its arguments turn up: the loop
    # spends its budget down this list in order. A counter rather than a
    # position in ``order``, which shifts as calls are appended.
    pending_seq_by_index: dict[int, int] = field(default_factory = dict)
    seq_by_key: dict[Any, int] = field(default_factory = dict)
    seq_counter: int = 0
    # Which call each id names, so a fragment repeating an id reaches its own
    # call even when a later call at that index is the one currently open.
    key_by_call_id: dict[str, Any] = field(default_factory = dict)
    # Resumable boundary scan per call, keyed the same as ``by_index``.
    scan_by_key: dict[Any, _BoundaryScan] = field(default_factory = dict)
    # Split-born calls whose object had not closed when they were forked off.
    # Reported only once it does: a stream cut short after "{\"a\":1}{" would
    # otherwise run the tool a second time on half an argument.
    open_tail_keys: set[Any] = field(default_factory = set)
    round: int = 0
    healed: list[dict[str, Any]] = field(default_factory = list)
    text: list[str] = field(default_factory = list)
    reasoning_extra: dict[str, Any] | None = None
    finish_reason: str | None = None
    # Results from tools the PROVIDER ran this turn, keyed by call id so a
    # repeated end event cannot record the same result twice.
    hosted_results: dict[str, dict[str, Any]] = field(default_factory = dict)

    def note_hosted_tool_event(self, event: Any) -> None:
        """Record a provider-side tool call carried on ``_toolEvent``.

        These reach the client as their own frames but are not part of the
        assistant message this loop replays, so the follow-up request would lose
        whatever the provider just produced. Unsloth's own events carry a
        top-level ``type``, so ``_toolEvent`` is unambiguously the provider's.

        Both halves matter: ``tool_end`` generally omits ``tool_name``, and for
        Gemini code execution the code that ran is only in the ``tool_start``
        arguments, so a result recorded alone is unlabelled.
        """
        if not isinstance(event, dict):
            return
        call_id = event.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            return
        kind = event.get("type")
        if kind not in ("tool_start", "tool_end"):
            return

        entry = self.hosted_results.setdefault(call_id, {})
        name = event.get("tool_name")
        if isinstance(name, str) and name:
            entry["name"] = name
        # The operation itself: Gemini's language and code, a search's query.
        # Merged across both halves because OpenAI opens an image generation
        # before it knows the prompt and only names it on the end event.
        arguments = _hosted_arguments_for_model(event.get("arguments"))
        if arguments:
            merged = dict(entry.get("arguments_obj") or {})
            merged.update(arguments)
            entry["arguments_obj"] = merged
            # Truncated with the same notice a result gets: Anthropic hands the
            # model's whole tool input through, so a file the code wrote lives
            # here and nowhere else, and a silent cut reads as the whole thing.
            entry["arguments"] = _truncate_for_model(
                json.dumps(merged, separators = (",", ":")),
                _HOSTED_ARGUMENT_MAX_CHARS,
                joiner = " ",
            )

        if kind == "tool_start":
            return
        result = event.get("result")
        if isinstance(result, str):
            # The call finished, even if it produced nothing: Gemini reports
            # code that printed nothing as an empty string. Recorded apart from
            # the result so that stays distinguishable from a stream that died
            # after the start. A non-string is a malformed frame, not an outcome.
            entry["ended"] = True
        if isinstance(result, str) and result.strip():
            if _carries_image_sentinel(result):
                # A Gemini plot with no stdout is nothing BUT the sentinel, so
                # stripping leaves an empty string and the entry looks empty.
                entry["produced_image"] = True
            # Same normalisation local results get: the frontend sentinels carry
            # a full data URI, and replaying one sends megabytes of base64. The
            # tool's name goes with it, as the local path passes it: only the
            # sandbox tools emit __FILES__, so a fetched page ending in a well
            # formed one keeps that line as the content it is.
            stripped = strip_result_for_model(result, entry.get("name"))
            if stripped.strip():
                entry["result"] = _truncate_for_model(stripped)
        if event.get("image_b64"):
            # image_generation reports an empty result and carries the picture
            # apart. Record that it happened rather than the bytes.
            entry["produced_image"] = True

    def hosted_replay_text(self) -> str:
        """The provider-run calls of this turn, as prose for the next request."""
        blocks: list[str] = []
        for entry in self.hosted_results.values():
            result = entry.get("result", "")
            produced_image = entry.get("produced_image")
            if not result and not produced_image and not entry.get("ended"):
                # A start with no outcome says only that something began.
                continue
            name = entry.get("name") or "tool"
            header = f"[{name} result]"
            arguments = entry.get("arguments")
            if arguments:
                header = f"[{name} {arguments}]"
            # A call that ended with nothing to show still has to appear, as the
            # same "(no output)" local tools report, so the model can tell the
            # code ran from it never having run at all.
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
                # A server that stamps index only on the opening fragment sends
                # the argument fragments bare. Treating those as a new call left
                # the real one with empty arguments and ran the tool anyway, so
                # continue the call that is already open instead.
                index = self.last_index if self.last_index is not None else len(self.order)
            call_id = raw_call.get("id")
            # continue whichever call owns this index now: index restarts at 0
            # for every tool round, so after a fork the bare argument fragments
            # belong to the newer call.
            key: Any = self.open_key_by_index.get(index, index)
            if isinstance(call_id, str) and call_id:
                # An id beats the latest-index mapping, which only exists to
                # place the fragments that carry no id: a fragment repeating an
                # id goes back to the call wearing it wherever that call now
                # sits, even when an id-less call opened at this index after it.
                # Matching only against the open call renamed that newer call
                # and gave it a second copy of the id.
                owner = self.key_by_call_id.get(call_id)
                if owner is not None:
                    key = owner
                elif self.by_index.get(key, {}).get("id"):
                    # Two distinct calls reported at the same index. Merging them
                    # concatenates their argument JSON into one unparseable blob
                    # and loses an intent, so key the second on its own id.
                    key = (index, call_id)
            # A closed object cannot take more content, so the next arguments to
            # reach a slot already holding one whole object belong to the next
            # parallel call. Forking on the accumulated text alone only catches
            # this once those arguments glue on, and a delta carrying an id
            # would claim the finished call and append to it (issue #9807).
            held = self.by_index.get(key)
            new_function = raw_call.get("function")
            new_arguments = (
                new_function.get("arguments") if isinstance(new_function, dict) else None
            )
            new_name = new_function.get("name") if isinstance(new_function, dict) else None
            # Whitespace after a closing brace is legal JSON and says nothing
            # about another call, so a provider that chunks it off on its own
            # must not open one.
            # An id, though, names a call outright, so one that is not the closed
            # call's own opens the next one even before its arguments arrive --
            # the conventional opening delta carries id and name with empty
            # arguments, and letting it land on the finished call would put the
            # next call's arguments there too.
            # A next call opens with the "{" of its own arguments object, so a
            # fragment that starts with anything else is not one. Forking on any
            # non-whitespace text cut where the scanner deliberately would not,
            # turning a stray scalar suffix into a second call and running the
            # tool twice.
            # An id at a closed slot opens the next call only when it also
            # names a different one. On its own, or repeating the name the slot
            # holds, it is that call's real id stamped late, and forking there
            # leaves the finished call under its provisional id beside an empty
            # second one.
            held_name_now = held["function"]["name"] if held is not None else ""
            id_names_another_call = bool(
                isinstance(call_id, str)
                and call_id
                and isinstance(new_name, str)
                and new_name
                and held_name_now
                # Not a prefix test: an id is strong evidence of its own call,
                # and a catalog holding both "web" and "web_search" would have
                # the second claim the first and glue their arguments together.
                # Only the same name, or none, reads as that call's id arriving
                # late.
                and new_name != held_name_now
            )
            opens_next_call = (
                bool(isinstance(new_arguments, str) and new_arguments.strip().startswith("{"))
                or id_names_another_call
            )
            # An id names its call, so a fragment repeating the id this slot
            # already holds continues it however complete its arguments look --
            # llama-server grows the name across deltas, and forking there gives
            # two calls one id, with the arguments on the abandoned name.
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
            # A name reaching a closed slot cannot be read yet: the same tool
            # called twice announces itself exactly as llama-server resends a
            # name it is still growing. So hold it until arguments say which
            # call it belongs to, rather than renaming the finished one.
            extra = raw_call.get("extra_content")
            # Trailing whitespace is legal JSON that belongs to the object just
            # closed, so the whitespace goes on the closed call. The name on such
            # a delta is still held rather than merged into that call: it is
            # either that call's name resent, which _take_parked discards when
            # the opening delta disagrees, or the next call's announced early,
            # and merging it gave the closed call "alphabeta" and the new call
            # no name at all, so neither ran.
            defers_to_next_call = slot_is_closed and not opens_next_call
            suppress_name = False
            suppress_extra = False
            if defers_to_next_call:
                if isinstance(new_name, str) and new_name:
                    # Held across deltas, so the two dialects the accumulator
                    # below reconciles have to be reconciled here too: a name
                    # streamed as "web" then "_search" must open its call as
                    # "web_search", not as whichever fragment arrived last.
                    pending_before = self.pending_name_by_index.get(index, "")
                    if index not in self.pending_seq_by_index:
                        self.pending_seq_by_index[index] = self._next_seq()
                    self.pending_name_by_index[index] = (
                        new_name
                        if new_name.startswith(pending_before)
                        else pending_before + new_name
                    )
                    suppress_name = True
                # Only once a name has announced the next call: metadata that
                # arrives alone belongs to the call that just closed, and
                # parking it there loses the signature outright when no further
                # call follows.
                if suppress_name and isinstance(extra, dict) and extra:
                    # Announced with the name, so it describes the call being
                    # announced. Merging it into the closed call leaves that
                    # call wearing another call's thoughtSignature and the new
                    # call with none, and native replay rejects both.
                    self.pending_extra_by_index[index] = {
                        **self.pending_extra_by_index.get(index, {}),
                        **extra,
                    }
                    suppress_extra = True
            if slot_is_closed and opens_next_call:
                self.split_seq += 1
                key = (index, "_split", self.split_seq)
            self.last_index = index
            self.open_key_by_index[index] = key
            if key not in self.by_index:
                parked_name, parked_extra = self._take_parked(index, held, new_name)
                born = {
                    "id": "",
                    "type": "function",
                    # The name a name-only delta parked for whichever call the
                    # arguments turned out to open, and the metadata that came
                    # with it.
                    "function": {"name": parked_name, "arguments": ""},
                }
                # The call takes the moment it was announced at, not the moment
                # its arguments arrived, so a call announced second is not run
                # third because another index opened in between.
                announced_at = self.pending_seq_by_index.pop(index, None)
                self.seq_by_key[key] = (
                    announced_at if announced_at is not None else self._next_seq()
                )
                if parked_extra:
                    born["extra_content"] = parked_extra
                self.by_index[key] = born
                # First-seen order, so a negative or out-of-order index cannot
                # reorder parallel calls against what the model actually sent.
                self.order.append(key)
            current = self.by_index[key]
            if isinstance(call_id, str) and call_id:
                current["id"] = call_id
                self.key_by_call_id.setdefault(call_id, key)
            # What the slot carried before this delta, so a fork below can hand
            # this delta's metadata to the call it actually closes.
            extra_before = current.get("extra_content")
            if isinstance(extra, dict) and extra and not suppress_extra:
                # Gemini 3 stows this call's thoughtSignature here, and the
                # native translator rejects a replayed functionCall without it.
                # Per call, so it cannot ride along on the delta-level slot.
                current["extra_content"] = {**current.get("extra_content", {}), **extra}
            function = raw_call.get("function")
            if isinstance(function, dict):
                # Two provider dialects, and picking either one alone breaks the
                # other. llama-server re-sends the whole name as it grows ("web"
                # then "web_search"), so appending yields "webweb_search".
                # OpenAI streams it in fragments ("web" then "_search"), so
                # assigning yields "_search". Both then fail the enabled-name
                # check and the call silently never runs. A fragment that already
                # starts with what we have is the whole name resent; anything
                # else continues it.
                fragment = function.get("name")
                name_before = current["function"]["name"]
                if isinstance(fragment, str) and fragment and not suppress_name:
                    if fragment.startswith(name_before):
                        current["function"]["name"] = fragment
                    else:
                        current["function"]["name"] = name_before + fragment
                if isinstance(function.get("arguments"), str):
                    current["function"]["arguments"] += function["arguments"]
                    if not (isinstance(call_id, str) and call_id):
                        # The id fork above cannot see this one: an id-less
                        # stream has no ids to differ, so appending glued two
                        # calls into one unparseable blob that then rides into
                        # the next request verbatim (issue #9807).
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
        """``_split_top_level_json_objects(text)``, resuming the scan for ``key``.

        The same answer as scanning from byte zero, at the cost of the bytes
        this call added rather than of the whole accumulation.
        """
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
        # Below rewrites this slot's arguments rather than extending them, so
        # the resumable scan no longer describes the string it was reading.
        self.scan_by_key.pop(key, None)
        # The slot keeps the object it opened with, under the name and id it
        # had. Nothing per-call rides along: extra_content carries this call's
        # thoughtSignature, and two calls claiming it is a rejected turn.
        current["function"]["arguments"] = segments[0]
        # A name arriving with this delta names the calls it opened, so the slot
        # goes back to its own. Without this the two dialects above merge them:
        # "alpha" then "gamma" at one index becomes "alphagamma", matching no
        # enabled tool and silently never running.
        born_name = incoming_name or name_before
        current["function"]["name"] = name_before or born_name
        # The metadata this delta carried belongs to the call this delta closes,
        # which is the last one, not to the object the slot already held. Gemini
        # checks the opaque signature against the exact call it is replayed on,
        # so leaving it here gets the follow-up rejected. Same placement as the
        # frontend split.
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
        # Later id-less fragments continue the last call, finished or not.
        self.open_key_by_index[index] = open_key

    def _take_parked(
        self, index: int, held: dict[str, Any] | None, opening_name: Any
    ) -> tuple[str, dict[str, Any] | None]:
        """The parked name and metadata to open a call with, if they are its.

        A parked name that repeats or extends the closed call's own is most
        likely that call's name resent, and is only kept because a second
        no-argument call to the same tool is indistinguishable from one. So
        when the delta that opens the call names it outright, that name wins:
        seeding "alpha_long" and merging "beta" onto it produced
        "alpha_longbeta", which matches no enabled tool and never runs.

        The metadata parked with that name follows the same decision. It was
        announced alongside the name, so once the name is read as the closed
        call's the metadata is the closed call's too; handing it to the new
        call gives that call another call's thought signature and leaves the
        closed one without its own, and the provider rejects both on replay.
        """
        parked = self.pending_name_by_index.pop(index, "")
        extra = self.pending_extra_by_index.pop(index, None)
        if not parked or not isinstance(opening_name, str) or not opening_name:
            return parked, extra
        held_name = held["function"]["name"] if held is not None else ""
        resent = bool(held_name) and (held_name.startswith(parked) or parked.startswith(held_name))
        if resent and not (opening_name.startswith(parked) or parked.startswith(opening_name)):
            if extra and held is not None:
                held["extra_content"] = {**held.get("extra_content", {}), **extra}
            # The moment goes with the announcement. This call was not the one
            # announced then, so it takes the moment its arguments arrived;
            # keeping the resend's would run it ahead of calls the stream
            # really did open first.
            self.pending_seq_by_index.pop(index, None)
            return "", None
        return parked, extra

    def _announced_but_unopened(self) -> list[tuple[int, dict[str, Any]]]:
        """Calls a name announced that no argument fragment ever opened.

        A tool that takes no parameters can be announced and then simply end,
        and ``_normalized_call`` already reads empty arguments as ``{}``, so
        dropping these runs one fewer tool than the model asked for.

        A name that repeats or extends the one the slot already holds is left
        alone: that is exactly how llama-server resends a name, or grows one,
        and it is indistinguishable from a second no-argument call to the same
        tool, so inventing a call there would run a tool twice off one request.
        Read rather than flushed, so a later argument fragment that does open
        the call still opens it, and this stops reporting it.
        """
        out: list[tuple[int, dict[str, Any]]] = []
        for index in sorted(self.pending_name_by_index):
            name = self.pending_name_by_index[index]
            if not name:
                continue
            held = self.by_index.get(self.open_key_by_index.get(index, index))
            held_name = held["function"]["name"] if held is not None else ""
            if held_name and (held_name.startswith(name) or name.startswith(held_name)):
                # No call is invented here, so metadata parked with that name
                # has nowhere else to go: it was announced on a delta that
                # turned out to be the closed call's name resent, which makes
                # it the closed call's, and dropping it costs that call its
                # thought signature on replay.
                extra = self.pending_extra_by_index.get(index)
                if extra and held is not None:
                    held["extra_content"] = {**held.get("extra_content", {}), **extra}
                continue
            call: dict[str, Any] = {
                "id": "",
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }
            extra = self.pending_extra_by_index.get(index)
            if extra:
                call["extra_content"] = dict(extra)
            out.append((self.pending_seq_by_index.get(index, self.seq_counter), call))
        return out

    def _call_is_finished(self, key: Any) -> bool:
        """Whether a call forked off an unfinished object has since closed it.

        Only ``length`` and ``content_filter`` mark a turn truncated, so a
        stream that stops after ``{"a":1}{`` looks complete; running the tool a
        second time on that lone brace is worse than dropping a call the model
        never finished writing.
        """
        if key not in self.open_tail_keys:
            return True
        closed, unfinished = _split_top_level_json_objects(
            self.by_index[key]["function"]["arguments"]
        )
        return bool(closed) and not unfinished

    def _next_seq(self) -> int:
        self.seq_counter += 1
        return self.seq_counter

    def calls(self, taken: set[str] | None = None) -> list[dict[str, Any]]:
        """Every call this turn produced, with ids unique across the whole run.

        ``taken`` carries the ids already used by earlier turns. A provider that
        restarts its numbering each turn, and the healer (which always mints
        call_0 first), would otherwise put two different results under one id in
        the conversation replayed upstream.
        """
        seen: set[str] = taken if taken is not None else set()
        out: list[dict[str, Any]] = []
        # Ordered by when the stream announced each call, so one announced
        # second is not run third because another index opened in between. Keyed
        # on the sequence number alone: two announcements can share a moment
        # only if one has no number yet, and comparing the calls themselves to
        # break that tie raises rather than sorting. Stable, so a shared number
        # keeps the order the calls were found in.
        numbered = [
            (self.seq_by_key.get(key, position), self.by_index[key])
            for position, key in enumerate(self.order)
            if self._call_is_finished(key)
        ] + self._announced_but_unopened()
        ordered = [call for _, call in sorted(numbered, key = lambda pair: pair[0])]
        for position, call in enumerate(ordered + list(self.healed)):
            normalized = _normalized_call(call, fallback_id = f"call_{self.round}_{position}")
            if normalized is None:
                continue
            if normalized["id"] in seen:
                # The client keyed the card it painted on the id the provider
                # streamed, so keep that one for the events aimed at the card.
                # Never replayed upstream: the conversation carries the
                # de-duplicated id, which is the whole point of the rename.
                normalized["stream_id"] = normalized["id"]
                # The renamed id is itself stored and replayed, so a single-shot
                # rename collides again on the next request. Counting up over a
                # finite ledger terminates and leaves the first attempt as is.
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


def _unrun_provenance(tool_name: str, round_id: int) -> dict[str, Any]:
    """Provenance for a hand-built unrun card; carries the MCP display name so a
    budget-exhausted or truncated MCP call never shows the internal server id."""
    provenance: dict[str, Any] = {"source": "local", "round_id": round_id}
    mcp = mcp_display_parts(tool_name)
    if mcp:
        provenance["mcp_server"] = mcp[0]
    return provenance


def _unrun_call_card(
    *, tool_name: str, tool_call_id: str, arguments: Any, result: str, provenance: dict[str, Any]
) -> list[str]:
    """The open/close pair for a call this loop announces but never runs.

    The close on its own is not enough. A call the provider streamed as a
    tool_calls delta already has a card, and the client reconciles both events
    onto it by id -- but a call the healer promoted out of TEXT was never
    streamed as a delta, so a lone tool_end names a card that does not exist and
    the adapter drops it: the user is told nothing at all. Opening the card
    first makes both cases end the same way, and keeps the invariant the loop is
    tested on, that every tool_end closes a tool_start.
    """
    return [
        _sse(
            {
                "type": "tool_start",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments if isinstance(arguments, dict) else {},
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


def _status_sse(text: str) -> str:
    """Tool badge text, in the shape the chat client already parses."""
    return _sse({"type": "tool_status", "content": text})


def _merge_usage(totals: dict[str, Any], usage: Any) -> None:
    """Sum one turn's usage into the running total.

    Per-turn usage is withheld while the loop runs and one summed chunk is sent
    at the end, so a multi-turn answer reports the same shape a single-turn one
    does instead of a burst of partial counts the client would have to add up.
    Detail sub-objects are summed too: reporting less than the same provider's
    plain stream would understate cost and cache hits.
    """
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


def _usage_chunk_line(model: str, totals: dict[str, Any]) -> str | None:
    if not totals:
        return None
    return _sse(
        {
            "id": "chatcmpl-external-tools",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [],
            "usage": totals,
        }
    )


def _is_usage_only(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    return "usage" in payload and isinstance(choices, list) and not choices


def _replayed_call_ids(conversation: list[dict[str, Any]]) -> set[str]:
    """Every tool-call id already in the history this run starts from.

    The healer restarts its counter every request, so a freshly minted call_0
    collides with a stripped call_0 replayed from history inside one upstream
    body. Seeding the ledger makes calls() rename the new one as it does any
    repeat within a run.
    """
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


def _append_user_turn(conversation: list[dict[str, Any]], content: str) -> None:
    """Append a user turn, merging into a trailing one so roles keep alternating.

    A turn whose only calls were no-ops appends no assistant message, so a bare
    append would leave two user turns in a row. Unlike the in-process loops this
    conversation is rendered by the provider, and a strict server rejects that.
    """
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
    """Join a pending ``next(gen)`` worker before its generator is closed.

    Cancelling the awaiting task does not stop the worker thread, and calling
    close() while next() is still running raises "generator already executing"
    and skips the generator's own cleanup. Setting the cancel flag lets a
    cancel-observing tool return, then the task is shielded until it finishes.
    """
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
) -> AsyncIterator[str]:
    """Stream a provider, execute requested Unsloth tools, continue to a final answer."""
    conversation = [dict(message) for message in run.messages]
    # Kept before the loop appends anything: this is the branch the request is on.
    request_branch = list(run.messages)
    remaining = policy.max_calls
    unlimited = remaining >= 9999
    session_id = run.session_id
    thread_id = run.thread_id
    tools = policy.tools
    tool_choice = run.tool_choice if run.tool_choice is not None else "auto"
    allowed_tool_names = _tool_names(tools)
    tool_call_timeout = policy.timeout
    permission_mode = policy.permission_mode
    confirm_tool_calls = policy.confirm_calls
    bypass_permissions = policy.bypass_permissions
    rag_scope = policy.rag_scope

    # The promotion allowlist is the selected catalog, never None: an
    # unrestricted parse re-opens markerless tool-call promotion.
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
    # Dedup, one-shot tracking and the force-final-answer transition are the same
    # ledger the local loops keep, so an external model cannot spend the budget
    # repeating one call and a terminal no-op still ends the loop.
    controller = ToolLoopController(
        tools = tools,
        auto_heal_tool_calls = policy.auto_heal is not False,
    )
    tool_hint = ", ".join(sorted(allowed_tool_names))
    reprompts = 0
    max_reprompts = MAX_ACT_REPROMPTS
    last_reprompt_text = ""
    provider_turns = 0
    used_call_ids: set[str] = _replayed_call_ids(conversation)
    spent_budget_passes = 0
    fruitless_turns = 0
    # One provider call per possible execution, plus headroom for the no-op,
    # nudge and final-answer passes that legitimately execute nothing. The
    # unlimited sentinel keeps its own budget rather than dropping to a smaller
    # fixed number: both local loops run "Max" for as many turns as the model
    # asks for, and fruitless_turns already ends a run that executes nothing, so
    # a lower bound here only cuts a productive run short with no final answer.
    max_provider_turns = max(1, remaining) + 2 * MAX_ACT_REPROMPTS + 4

    while not cancel_event.is_set():
        if provider_turns >= max_provider_turns:
            # Reached only by a model that keeps asking for tools it cannot run
            # (all disabled, or the budget is gone). Executions are already
            # capped; this caps the asking, so the conversation cannot grow
            # without bound when nothing ever executes.
            break
        provider_turns += 1
        turn = _Turn(round = provider_turns)
        healer = StreamToolCallHealer(heal_names, tools) if heal_names else None

        active_tools = controller.active_tools()
        tools_available = (
            tool_choice != "none" and bool(active_tools) and (unlimited or remaining > 0)
        )
        # Withdrawing the catalog and pinning "none" together: this path owns the
        # tool surface (the route withholds enabled_tools), so there are no
        # provider-hosted builtins left for "none" to revoke, and saying it
        # explicitly stops a model from calling a tool it was just denied.
        turn_tool_choice = tool_choice if tools_available else "none"
        # A forced choice applies until the model actually calls something; the
        # result follow-up must be free to answer in prose.
        if executed_any and turn_tool_choice not in ("auto", "none"):
            turn_tool_choice = "auto"

        generator = transport.stream(
            messages = conversation,
            tools = active_tools if tools_available else None,
            tool_choice = turn_tool_choice,
            cancel_event = cancel_event,
        )
        try:
            async for line in generator:
                if _is_done_sentinel(line):
                    # Every turn ends with one. Relaying it mid-loop tells a
                    # spec-compliant client the response is over, and it stops before
                    # the tool cards and the real answer. The route emits the final one.
                    continue
                # This loop writes its tool cards, badges and the approval
                # handshake onto the same stream the provider's chunks are
                # relayed on, and the client tells them apart only by shape. A
                # provider's copy of that vocabulary would therefore paint a card
                # for a tool that never ran, so strip it before anything below
                # can relay it. Skipped for a transport that already did it at
                # the point the raw bytes arrived: everything reaching here from
                # one of those is this server's own frame, and stripping it drops
                # a retained hosted tool's result after the provider billed it.
                if not transport_sanitizes:
                    sanitized = sanitize_provider_sse_line(line)
                    if sanitized is None:
                        continue
                    line = sanitized
                payload = _chunk_payload(line)
                if payload is None:
                    yield line
                    continue
                # OpenRouter and friends resolve a routing alias to a concrete
                # model and name it on every chunk. That id is more specific than
                # the one the request asked for, so let it win for the summed
                # usage chunk this loop emits once the answer ends.
                upstream_model = payload.get("model")
                if isinstance(upstream_model, str) and upstream_model:
                    model_name = upstream_model
                if "usage" in payload:
                    _merge_usage(usage_totals, payload.get("usage"))
                    if _is_usage_only(payload):
                        # Withheld: one summed chunk is sent once the loop ends, so a
                        # multi-turn answer does not report a burst of partial counts.
                        continue
                    # Some providers hang usage off a chunk that also carries a
                    # choice, which cannot be withheld wholesale without losing
                    # the content. Drop just the usage: it is already in the
                    # totals, and leaving it here makes a client that sums
                    # chunks count this turn twice.
                    payload.pop("usage", None)
                    line = "data: " + json.dumps(payload, separators = (",", ":"))
                choices = payload.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else {}
                if not isinstance(choice, dict):
                    yield line
                    continue

                delta = choice.get("delta")
                delta = delta if isinstance(delta, dict) else {}
                content = delta.get("content")
                raw_calls = delta.get("tool_calls")
                extra = delta.get("extra_content")
                if isinstance(extra, dict):
                    turn.reasoning_extra = extra
                turn.note_hosted_tool_event(payload.get("_toolEvent"))
                if isinstance(choice.get("finish_reason"), str):
                    turn.finish_reason = choice["finish_reason"]

                if isinstance(raw_calls, list) and raw_calls:
                    if healer is not None and not healer.dormant:
                        # Grammar mode worked; stop second-guessing the text stream.
                        # Whatever was being held was ordinary prose after all, so it
                        # has to reach the client, not just the conversation replay.
                        for kind, value in healer.structured_tool_call_seen():
                            if kind == "text" and value:
                                turn.text.append(value)
                                yield _sse({"choices": [{"index": 0, "delta": {"content": value}}]})
                    turn.merge_structured(raw_calls)

                if healer is None or healer.dormant or not isinstance(content, str) or not content:
                    plain = _delta_text(content)
                    if plain:
                        turn.text.append(plain)
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
                    # Nothing was held back, which is the case for almost every chunk
                    # of ordinary prose. Relay the provider's own bytes rather than
                    # paying a re-encode per chunk to reproduce them.
                    yield line
                    continue
                # Withholding everything is normal mid-block. Only drop the chunk when
                # it carries nothing else worth relaying.
                if visible or turn.finish_reason is not None or len(delta) > 1:
                    yield _rewrite_content(payload, choice, visible)

            if healer is not None:
                for kind, value in healer.finalize():
                    if kind == "text":
                        if value:
                            turn.text.append(value)
                            yield _sse({"choices": [{"index": 0, "delta": {"content": value}}]})
                    elif kind == "tool_call":
                        turn.healed.append(value)

        finally:
            # Release the upstream response now rather than leaving it to the
            # async-generator finalisation hook, which runs a tick or more after
            # the route has already closed this loop.
            aclose = getattr(generator, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except (RuntimeError, GeneratorExit):
                    pass

        # Both of these mean the turn ended before the model finished saying what
        # it wanted: "length" hit the token ceiling, "content_filter" had the
        # output cut by the provider's own filter. Either way a call collected so
        # far may be half-written, so it is described rather than run. "stop" is
        # deliberately not in this set: llama.cpp and vLLM routinely finish a
        # perfectly good tool call with it, and refusing those would disable
        # tool calling on exactly the self-hosted servers this path exists for.
        truncated = turn.finish_reason in ("length", "content_filter")
        if truncated and healer is not None and turn.healed:
            # A call cut off at the token limit must not run: its arguments can
            # be half-written and the model never finished saying what it wanted.
            # But promotion is destructive -- the healer already removed the
            # markup span from the text relayed above -- so discarding the call
            # on its own would take the sentence that introduced it with it, and
            # the user would be left with a stub answer, no card, and no way to
            # tell that anything was attempted. Give the exact removed span back
            # instead. It is the same verbatim flush the healer performs for a
            # block that turns out not to be a declared call, and only the healer
            # can supply it: nothing else in the loop keeps the raw bytes, and
            # re-encoding the parsed call would print something the model never
            # wrote. Released at the end of the turn rather than in document
            # order because "length" is only known once the turn has ended.
            for healed_call in turn.healed:
                span = healer.promoted_source(healed_call.get("id", ""))
                if not span:
                    continue
                turn.text.append(span)
                yield _sse({"choices": [{"index": 0, "delta": {"content": span}}]})
        if truncated:
            # The other half of the same problem. A call the provider streamed as
            # a tool_calls delta was relayed as it arrived, so the client already
            # has a card for it, and refusing to run it leaves that card open for
            # the rest of the response. Close it the way every other unrun call
            # is closed. Structured only: a healed call was never streamed, and
            # the span released just above is what tells the user about that one.
            for raw_call in turn.by_index.values():
                truncated_id = raw_call.get("id")
                function = raw_call.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if not isinstance(truncated_id, str) or not truncated_id:
                    continue
                if not isinstance(name, str) or not name:
                    # Not enough of the call arrived to name a tool, so there is
                    # no card of ours to close.
                    continue
                for card_line in _unrun_call_card(
                    tool_name = name,
                    tool_call_id = truncated_id,
                    # The arguments are cut off mid-write, so there is nothing
                    # well formed to show; the result says what happened.
                    arguments = {},
                    result = _TOOL_TRUNCATED,
                    provenance = _unrun_provenance(name, round_id + 1),
                ):
                    yield card_line
        # tool_choice "none" is an instruction, and a provider that emits a call
        # anyway has not been authorized to run one. Withdrawing the catalog on
        # the way out is not enough on its own: Deep Research sets "none" exactly
        # so the scraped web text in its prompts cannot reach python or terminal,
        # so a naive or compromised endpoint echoing a call back must not be able
        # to execute it here.
        calls = [] if (truncated or tool_choice == "none") else turn.calls(used_call_ids)
        if not calls:
            # No tool this turn. A model that only said what it was about to do
            # gets one nudge to actually do it, the same recovery the local loops
            # give a stalled small model, then the answer stands as written.
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
                    # A hosted tool did run, the model just did not go on to ask
                    # for a local one. The replay below never happens on this
                    # path, so the reprompted request would be told to continue
                    # from output it can no longer see.
                    stalled_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": (
                            f"{visible_answer}\n\n{stalled_hosted}"
                            if visible_answer
                            else stalled_hosted
                        ),
                    }
                    if turn.reasoning_extra:
                        # Gemini 3 stows the text part's thoughtSignature here
                        # and its translator pins it back on from this field
                        # alone, so a turn replayed without it is rejected.
                        stalled_message["extra_content"] = turn.reasoning_extra
                    append_assistant_turn(
                        conversation,
                        stalled_message,
                        # A resumed partial is the same turn as what the model
                        # just added, so merge rather than append: appending
                        # puts a turn boundary mid-sentence.
                        continue_final_message = run.continue_final_message,
                    )
                _append_user_turn(conversation, reprompt_to_act_message(tool_hint))
                continue
            break

        round_id += 1
        assistant_tool_calls: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        noop_messages: list[dict[str, Any]] = []
        turn_executed_real_tool = False

        for call in calls:
            if cancel_event.is_set():
                break
            if not unlimited and remaining <= 0:
                # Budget spent. Answer the model so it stops asking, but never
                # execute: the cap is a safety limit, not a hint to the provider.
                for card_line in _unrun_call_card(
                    tool_name = call["function"]["name"],
                    tool_call_id = call.get("stream_id") or call["id"],
                    arguments = call.get("arguments"),
                    result = _TOOL_BUDGET_EXHAUSTED,
                    provenance = _unrun_provenance(call["function"]["name"], round_id),
                ):
                    yield card_line
                # The result below has to be replayed with its call: only the
                # call that spent the last slot reaches assistant_tool_calls
                # further down, so this one would arrive as an orphan
                # role="tool" message and OpenAI, Anthropic and Gemini all
                # reject that history instead of answering.
                exhausted_call: dict[str, Any] = {
                    "id": call["id"],
                    "type": "function",
                    # Copied: the normalized call also carries a parsed
                    # arguments dict that must not reach the provider.
                    "function": dict(call["function"]),
                }
                exhausted_extra = call.get("extra_content")
                if isinstance(exhausted_extra, dict) and exhausted_extra:
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
            decision = controller.prepare_call(call)
            # The frontend groups a round's reasoning by this id
            # (codexLocalToolRoundId), so every tool card the loop emits has to
            # carry it, not just the budget-exhausted one built by hand above.
            decision.provenance["round_id"] = round_id
            if not decision.should_execute:
                completion = controller.record_noop(decision)
                noop_messages.append(completion.model_message())
                # The provider's own tool_calls delta for this call was relayed
                # verbatim while it streamed and the client painted a card from
                # it. Nothing else closes that card, so without a terminal event
                # it spins for the rest of the answer and then reads as a tool
                # that ran and returned nothing. Keyed on the streamed id, since
                # a repeated call is exactly the one this loop renames above.
                #
                # Only for a tool the user DID enable. A call for something
                # outside the catalog is not a tool of Unsloth's that declined to
                # run, it is a name this install never offered, and giving it a
                # card would advertise a tool the user switched off. That one is
                # answered in the conversation only.
                if decision.action == "disabled":
                    continue
                for card_line in _unrun_call_card(
                    tool_name = decision.tool_name,
                    tool_call_id = call.get("stream_id") or decision.tool_call_id,
                    arguments = decision.arguments,
                    result = _TOOL_SKIPPED.get(decision.action, "Unsloth did not run this call."),
                    provenance = decision.provenance,
                ):
                    yield card_line
                continue
            assistant_call = decision.as_assistant_tool_call()
            call_extra = call.get("extra_content")
            if isinstance(call_extra, dict) and call_extra:
                # Replayed verbatim: Gemini 3 validates the signature that came
                # back with this exact call.
                assistant_call["extra_content"] = call_extra
            assistant_tool_calls.append(assistant_call)

            name = decision.tool_name
            arguments = decision.arguments
            call_id = decision.tool_call_id
            needs_confirmation = (
                confirm_tool_calls and not bypass_permissions and permission_mode != "off"
            )
            if needs_confirmation and permission_mode == "auto":
                needs_confirmation = is_high_risk_tool_call(name, arguments)
            approval_id = new_approval_id() if needs_confirmation else ""
            decision_slot = (
                begin_tool_decision(session_id, approval_id) if needs_confirmation else None
            )

            start_event = decision.tool_start_event()
            start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirmation
            denied = False
            try:
                # A gated call has not started, so it must not read as running.
                yield _status_sse(
                    awaiting_approval_status(name) if needs_confirmation else decision.status_text
                )
                yield _sse(start_event)
                verdict = None
                if decision_slot is not None:
                    waiter = asyncio.ensure_future(
                        asyncio.to_thread(
                            wait_tool_decision, decision_slot, approval_id, cancel_event
                        )
                    )
                    try:
                        # Hold the first keepalive back so the gated card is flushed
                        # on its own write and the Allow / Deny buttons paint before
                        # the stream blocks waiting for the answer.
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
                yield _sse(
                    {
                        "type": "tool_end",
                        "tool_name": name,
                        "tool_call_id": call_id,
                        "result": TOOL_REJECTED_MESSAGE,
                        "provenance": decision.provenance,
                    }
                )
                denied_message: dict[str, Any] = {
                    "role": "tool",
                    "name": name,
                    "content": TOOL_REJECTED_MESSAGE,
                }
                if call_id:
                    denied_message["tool_call_id"] = call_id
                tool_messages.append(denied_message)
                # A denial is an answer, not a stall: never nudge after one.
                reprompts = max_reprompts
                continue

            def _invoke(output_callback: Any, call = decision) -> str:
                kwargs: dict[str, Any] = {
                    "cancel_event": cancel_event,
                    "timeout": None if tool_call_timeout >= 9999 else tool_call_timeout,
                    "session_id": session_id,
                    "thread_id": thread_id,
                    "rag_scope": rag_scope,
                    "disable_sandbox": bypass_permissions,
                }
                # Provider loops share the local catalogue selector, so
                # search_conversation is advertised here too once a thread has an archive
                # and needs the same branch: the stored rows are the whole DAG, and Retry
                # leaves the replaced response in them.
                if accepts_kwarg(execute_tool, "conversation_branch"):
                    kwargs["conversation_branch"] = request_branch
                # And a budget, so the tool's clamp is not skipped. Unsloth cannot measure
                # an external model's window, and a custom OpenAI-compatible endpoint can
                # be a small local server, so a model-chosen 8 chunks is roughly 4K tokens
                # replayed on every later call. Unmeasurable means one recall's worth.
                # Explicitly unknowable, not absent: this request is served by an
                # external provider, so the resident GGUF's window says nothing about
                # what it can hold. 0 keeps the default page cap instead of inheriting it.
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
                return execute_tool(call.tool_name, call.arguments, **kwargs)

            # The same wrapper the local loops run tools through: live stdout for
            # the card, and a heartbeat so a long call cannot idle the stream out.
            tool_stream = stream_tool_execution(
                _invoke,
                tool_name = name,
                tool_call_id = call_id,
                cancel_event = cancel_event,
            )
            outcome: dict[str, Any] = {}
            step_task: Any = None
            try:
                while True:
                    if cancel_event.is_set():
                        # A tool that does not watch the cancel event would keep
                        # producing heartbeats and hold the answer open forever.
                        # Stop asking for them and let the drain below join the
                        # worker under its own bounded timeout.
                        break
                    step_task = asyncio.create_task(
                        asyncio.to_thread(_advance_tool_stream, tool_stream, outcome)
                    )
                    # wait, not await: cancelling this coroutine must leave the
                    # worker pending so the drain below can still join it.
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
                    # Stopped before the tool returned. Defaulting to "" here
                    # would record a successful empty result and paint a normal
                    # tool_end, so the transcript would claim a tool ran and
                    # produced nothing, when in truth it was abandoned partway
                    # and its side effects may already have happened.
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
            # Counted whether or not the tool succeeded: a failing call has
            # already done its work (and possibly its side effects), so letting
            # it run for free would put the budget past max_calls. Counted per
            # call rather than per turn, so parallel calls each spend one.
            if not unlimited:
                remaining -= 1
            turn_executed_real_tool = True
            executed_any = True
            # Opens the post-tool phase; carried-over stall text would eat its nudge.
            last_reprompt_text = ""
            yield _sse(completion.tool_end_event())
            tool_messages.append(completion.tool_message())

        # An empty status clears the badge between iterations.
        yield _status_sse("")

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            # Markup never replays: the call is carried structurally below.
            "content": strip_tool_markup(
                "".join(turn.text), final = True, enabled_tool_names = allowed_tool_names
            ),
        }
        hosted_text = turn.hosted_replay_text()
        if hosted_text:
            # A tool the provider ran itself this turn. Its output went to the
            # client as its own frame but is not otherwise part of this message,
            # so the follow-up would answer from the local results alone.
            # Replayed as text, not native items: the shape differs per provider
            # (Gemini codeExecutionResult, an OpenAI image call), while every
            # provider can read its own prior turn's prose.
            assistant_message["content"] = (
                f"{assistant_message['content']}\n\n{hosted_text}"
                if assistant_message["content"]
                else hosted_text
            )
        if turn.reasoning_extra:
            assistant_message["extra_content"] = turn.reasoning_extra
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls
        if assistant_message["content"] or assistant_tool_calls:
            # Merges into a resumed partial so a continued tool turn stays one message.
            append_assistant_turn(
                conversation,
                assistant_message,
                continue_final_message = run.continue_final_message,
            )
        conversation.extend(tool_messages)
        # Deferred to after the results so a no-op never splits a call from them,
        # and merged into a trailing user turn so the roles keep alternating.
        _append_user_turn(
            conversation,
            "\n\n".join(dict.fromkeys(message["content"] for message in noop_messages)),
        )

        if turn_executed_real_tool:
            max_reprompts = _MAX_POST_TOOL_REPROMPTS
            reprompts = 0
            fruitless_turns = 0
        else:
            fruitless_turns += 1
            if fruitless_turns >= _MAX_FRUITLESS_TURNS:
                # It asked for tools twice running and none could run. One more
                # pass would only repeat the exchange, so stop asking.
                break
        if remaining <= 0 and not unlimited and not controller.force_final_answer:
            if spent_budget_passes:
                # It was already told the budget is gone and asked for a tool
                # again anyway. One more pass to let it answer, then stop rather
                # than trading turns with it.
                break
            spent_budget_passes += 1
            # The catalog is gone from here on, so say why rather than letting the
            # next pass ask for a tool that is no longer offered.
            _append_user_turn(conversation, _BUDGET_EXHAUSTED_NUDGE)

    usage_line = _usage_chunk_line(model_name, usage_totals)
    if usage_line is not None:
        yield usage_line
