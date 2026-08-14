# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio-owned tool execution loop, shared by every external provider transport.

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
"""

from __future__ import annotations

import asyncio
import json
import threading

from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any, Protocol

from core.inference.chat_template_helpers import append_assistant_turn
from core.inference.passthrough_healing import StreamToolCallHealer, heal_gate
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
)
from core.inference.tool_stream_exec import (
    TOOL_HEARTBEAT_INTERVAL_S,
    accepts_output_callback,
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
    "Studio did not execute this tool call because the per-message tool-call limit was reached. "
    "Continue with the available results and answer without calling another tool."
)

_TOOL_DISABLED = "Studio did not execute this tool call because the tool is disabled."

_TOOL_CANCELLED = (
    "Studio stopped this tool call before it returned, so there is no result. "
    "The tool may have already done part of its work."
)

_TOOL_TRUNCATED = (
    "Studio did not execute this tool call because the provider stopped mid-call at its "
    "output limit."
)

# Card text for a call the controller skipped. The client already painted a card
# from the provider's own tool_calls delta, so it needs a short result; the long
# model-facing nudge stays in the conversation.
_TOOL_SKIPPED = {
    "duplicate": "Studio did not run this call because an identical one had already completed.",
    "disabled": _TOOL_DISABLED,
    "render_html_repeat": "Studio did not run this call because render_html already ran.",
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
        # The id is Studio's correlation key, not the model's contract. Several
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

    # Whether the transport already stripped Studio's control vocabulary from
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


@dataclass
class _Turn:
    """Accumulated state for one provider turn."""

    by_index: dict[Any, dict[str, Any]] = field(default_factory = dict)
    order: list[Any] = field(default_factory = list)
    last_index: int | None = None
    round: int = 0
    healed: list[dict[str, Any]] = field(default_factory = list)
    text: list[str] = field(default_factory = list)
    reasoning_extra: dict[str, Any] | None = None
    finish_reason: str | None = None

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
            key: Any = index
            if isinstance(call_id, str) and call_id:
                open_id = self.by_index.get(index, {}).get("id")
                if open_id and open_id != call_id:
                    # Two distinct calls reported at the same index. Merging them
                    # concatenates their argument JSON into one unparseable blob
                    # and loses an intent, so key the second on its own id.
                    key = (index, call_id)
            self.last_index = index
            if key not in self.by_index:
                self.by_index[key] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
                # First-seen order, so a negative or out-of-order index cannot
                # reorder parallel calls against what the model actually sent.
                self.order.append(key)
            current = self.by_index[key]
            if isinstance(call_id, str) and call_id:
                current["id"] = call_id
            extra = raw_call.get("extra_content")
            if isinstance(extra, dict) and extra:
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
                if isinstance(fragment, str) and fragment:
                    accumulated = current["function"]["name"]
                    if fragment.startswith(accumulated):
                        current["function"]["name"] = fragment
                    else:
                        current["function"]["name"] = accumulated + fragment
                if isinstance(function.get("arguments"), str):
                    current["function"]["arguments"] += function["arguments"]

    def calls(self, taken: set[str] | None = None) -> list[dict[str, Any]]:
        """Every call this turn produced, with ids unique across the whole run.

        ``taken`` carries the ids already used by earlier turns. A provider that
        restarts its numbering each turn, and the healer (which always mints
        call_0 first), would otherwise put two different results under one id in
        the conversation replayed upstream.
        """
        seen: set[str] = taken if taken is not None else set()
        out: list[dict[str, Any]] = []
        for position, call in enumerate(
            [self.by_index[key] for key in self.order] + list(self.healed)
        ):
            normalized = _normalized_call(call, fallback_id = f"call_{self.round}_{position}")
            if normalized is None:
                continue
            if normalized["id"] in seen:
                # The client keyed the card it painted on the id the provider
                # streamed, so keep that one for the events aimed at the card.
                # Never replayed upstream: the conversation carries the
                # de-duplicated id, which is the whole point of the rename.
                normalized["stream_id"] = normalized["id"]
                normalized["id"] = f"{normalized['id']}_{self.round}_{position}"
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
    """Stream a provider, execute requested Studio tools, continue to a final answer."""
    conversation = [dict(message) for message in run.messages]
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
    used_call_ids: set[str] = set()
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
                    provenance = {"source": "local", "round_id": round_id + 1},
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
                and not controller.force_final_answer
                and reprompts < max_reprompts
                and is_short_intent_without_action(visible_answer)
                and not is_reprompt_repeat(visible_answer, last_reprompt_text)
            ):
                reprompts += 1
                last_reprompt_text = visible_answer
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
                    provenance = {"source": "local", "round_id": round_id},
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
                # outside the catalog is not a tool of Studio's that declined to
                # run, it is a name this install never offered, and giving it a
                # card would advertise a tool the user switched off. That one is
                # answered in the conversation only.
                if decision.action == "disabled":
                    continue
                for card_line in _unrun_call_card(
                    tool_name = decision.tool_name,
                    tool_call_id = call.get("stream_id") or decision.tool_call_id,
                    arguments = decision.arguments,
                    result = _TOOL_SKIPPED.get(decision.action, "Studio did not run this call."),
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
                if accepts_output_callback(execute_tool):
                    kwargs["output_callback"] = output_callback
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
