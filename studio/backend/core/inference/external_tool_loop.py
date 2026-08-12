# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local agentic tool loop over a self-hosted OpenAI-compatible provider.

vLLM, Ollama, llama.cpp and generic custom servers do the model half of
function calling but ship no tool runtime: they emit ``tool_calls`` and stop.
This module runs Unsloth's own runtime for them. It advertises the local tool
schemas on ``/v1/chat/completions``, executes the calls the server asks for,
appends the results, and re-prompts until the model answers. The SSE it emits
matches what the GGUF and safetensors loops emit, so the frontend renders
identical tool cards.

Hosted providers are deliberately excluded. OpenAI, Anthropic, Gemini,
OpenRouter and Kimi run their own server-side tools, already wired through
``enabled_tools`` in ``external_provider.py``; sending them a local tool
catalog would duplicate or conflict with those. ``search_knowledge_base``
(RAG) is excluded everywhere: retrieval stays local-only.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from typing import Any, AsyncGenerator, Callable, Iterable, Optional, Sequence

from loggers import get_logger

from core.inference.tool_call_parser import (
    MAX_ACT_REPROMPTS,
    NUDGE_TOOL_CALLS_STATUS,
    TOOL_XML_SIGNALS,
    is_reprompt_repeat,
    is_short_intent_without_action,
    parse_tool_calls_from_text,
    reprompt_to_act_message,
    strip_tool_markup,
)
from core.inference.tool_loop_controller import (
    ToolLoopController,
    append_deferred_nudges,
    awaiting_approval_status,
    tool_event_provenance,
)
from core.inference.tool_stream_exec import (
    TOOL_HEARTBEAT_INTERVAL_S,
    accepts_output_callback,
    stream_tool_execution,
)
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)

logger = get_logger(__name__)


# these self-hosted servers ship no tool runtime, so ours has nothing to collide with.
LOCAL_TOOL_LOOP_PROVIDER_TYPES = frozenset({"vllm", "ollama", "llama_cpp", "custom"})

# search_knowledge_base and render_html are excluded: RAG and Canvas stay local-only.
LOCAL_TOOL_LOOP_TOOL_NAMES = ("web_search", "python", "terminal")

# matches the local loops' default budget (generate_chat_completion_with_tools).
DEFAULT_MAX_TOOL_ITERATIONS = 25

# only a large streamed payload earns an early card, as in llama_cpp.py.
PROVISIONAL_ARGS_MIN_CHARS = 256

# a stall after a tool ran costs a re-run on retry, so the local loops allow one.
MAX_POST_TOOL_REPROMPTS = 1

# sse comment the route writes while a tool blocks; keeps proxies from idling out.
_SSE_KEEPALIVE = ": keep-alive"

# delay before the first approval keepalive so it cannot coalesce with the gated card.
TOOL_APPROVAL_FLUSH_DELAY_S = 0.05

# text-form calls are not grammar-bounded the way delta.tool_calls are, so one
# runaway turn is capped here as it is in the local loops.
MAX_TEXT_TOOL_CALLS_PER_TURN = 8

# verbatim from the local loops: the last pass answers instead of asking for more tools.
BUDGET_EXHAUSTED_NUDGE = (
    "You have used all available tool calls. Based on everything you have found "
    "so far, provide your final answer now. Do not call any more tools."
)

# longest partial marker worth holding back at a delta boundary.
_MAX_SIGNAL_PREFIX = max(len(signal) for signal in TOOL_XML_SIGNALS) - 1


def _first_signal_index(text: str) -> Optional[int]:
    """Offset of the earliest tool marker in ``text``, or None."""
    found = [text.index(signal) for signal in TOOL_XML_SIGNALS if signal in text]
    return min(found) if found else None


def _holdback_len(text: str) -> int:
    """How much of the tail could still grow into a tool marker."""
    for size in range(min(len(text), _MAX_SIGNAL_PREFIX), 0, -1):
        suffix = text[-size:]
        if any(signal.startswith(suffix) for signal in TOOL_XML_SIGNALS):
            return size
    return 0


def local_tool_loop_supported(provider_type: Optional[str]) -> bool:
    """Whether this provider type may run the local tool loop."""
    if not provider_type or provider_type not in LOCAL_TOOL_LOOP_PROVIDER_TYPES:
        return False
    from core.inference.providers import get_provider_info

    return bool((get_provider_info(provider_type) or {}).get("supports_tool_calling"))


def select_local_tool_names(enabled_tools: Optional[Iterable[str]]) -> list[str]:
    """Filter a request's ``enabled_tools`` down to what this loop may run.

    Order follows ``LOCAL_TOOL_LOOP_TOOL_NAMES`` so the advertised catalog is
    stable regardless of the order the client listed its pills in.
    """
    if not enabled_tools:
        return []
    requested = {str(name) for name in enabled_tools}
    return [name for name in LOCAL_TOOL_LOOP_TOOL_NAMES if name in requested]


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}"


def _status_sse(text: str) -> str:
    """Tool badge text, already in the ``tool_status`` shape chat-api.ts parses."""
    return _sse({"type": "tool_status", "content": text})


def _first_delta(chunk: dict[str, Any]) -> dict[str, Any]:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    delta = choices[0].get("delta")
    return delta if isinstance(delta, dict) else {}


def _merge_usage(totals: dict[str, int], usage: Any) -> None:
    """Sum the countable fields of one turn's usage block into ``totals``."""
    if not isinstance(usage, dict):
        return
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        if isinstance(value, int):
            totals[field] = totals.get(field, 0) + value


class _TurnAccumulator:
    """Split one provider turn's SSE into client output and tool-call state.

    ``feed`` returns the lines to forward. ``tool_calls`` fragments are
    withheld: this loop executes them and emits its own ``tool_start`` /
    ``tool_end`` events, so forwarding the raw fragments would draw a second,
    never-resolved tool card. The terminal ``finish_reason`` chunk is withheld
    too, because the turn is not the response's last one whenever a tool call
    follows; its content, if any, is still forwarded.

    A call whose arguments are long enough opens a provisional card and streams
    its argument text, so a slow ``python`` payload renders while the model is
    still writing it. Same rule and same events as the GGUF loop.

    Content is gated on the way out. A model that writes its call as text rather
    than as ``delta.tool_calls`` would otherwise leak ``<tool_call>`` markup into
    the answer, so everything from the first marker on is withheld: the loop
    parses it into a real call, or releases it when it turns out to be prose.
    """

    def __init__(
        self,
        *,
        provisional_tool_names: frozenset[str] = frozenset(),
    ) -> None:
        self.text: str = ""
        self.shown: str = ""
        self.pending: str = ""
        self.suppressed_tail: str = ""
        self.markup_seen: bool = False
        self.reasoning: str = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: Optional[str] = None
        self.failed: bool = False
        self.usage: dict[str, int] = {}
        self.provisional_cards: dict[str, str] = {}
        self._provisional_tool_names = provisional_tool_names
        self._args_streamed: set[str] = set()
        self._final_chunk: Optional[dict[str, Any]] = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def final_chunk_line(self) -> Optional[str]:
        """The withheld terminal chunk, for the response's last turn."""
        if self._final_chunk is None:
            return None
        return _sse(self._final_chunk)

    def feed(self, line: str) -> list[str]:
        if not line.startswith("data:"):
            return []
        data_str = line[len("data:") :].strip()
        if not data_str or data_str == "[DONE]":
            return []
        try:
            chunk = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(chunk, dict):
            return []
        if "error" in chunk:
            self.failed = True
            return [line]

        _merge_usage(self.usage, chunk.pop("usage", None))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            # usage-only chunk, folded into the running totals and re-emitted at the end.
            return []
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = _first_delta(chunk)
        raw_tool_calls = delta.pop("tool_calls", None)
        extra: list[str] = []
        if isinstance(raw_tool_calls, list):
            extra = self._absorb_tool_calls(raw_tool_calls)
        # ollama names the thinking channel `reasoning`; the local loops and the
        # frontend both speak `reasoning_content`, so normalize it on the way out
        # or the thinking block never renders.
        reasoning = delta.pop("reasoning", None)
        if isinstance(reasoning, str) and reasoning and not delta.get("reasoning_content"):
            delta["reasoning_content"] = reasoning
        thinking = delta.get("reasoning_content")
        if isinstance(thinking, str) and thinking:
            self.reasoning += thinking
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.text += content
            content = self._gate_content(content)
        else:
            content = ""
        if "content" in delta:
            delta["content"] = content
            chunk["choices"][0]["delta"] = delta

        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            self.finish_reason = finish_reason
            terminal = copy.deepcopy(chunk)
            terminal["choices"][0]["delta"] = {}
            self._final_chunk = terminal
            if not content:
                return extra
            # the visible text ships now, the terminal marker only if this turn ends the response.
            carried = copy.deepcopy(chunk)
            carried["choices"][0]["finish_reason"] = None
            return extra + [_sse(carried)]

        # a chunk left with nothing to render (tool_calls only, or held-back markup).
        if not content and not any(delta.get(key) for key in delta):
            return extra
        return extra + [_sse(chunk)]

    def _gate_content(self, text: str) -> str:
        """Return the part of ``text`` safe to show; hold or drop tool markup."""
        if self.markup_seen:
            self.suppressed_tail += text
            return ""
        self.pending += text
        marker = _first_signal_index(self.pending)
        if marker is not None:
            self.markup_seen = True
            flushed, self.suppressed_tail = self.pending[:marker], self.pending[marker:]
            self.pending = ""
        else:
            # a tail that could still grow into a marker waits for the next delta.
            hold = _holdback_len(self.pending)
            flushed = self.pending[: len(self.pending) - hold] if hold else self.pending
            self.pending = self.pending[len(self.pending) - hold :] if hold else ""
        self.shown += flushed
        return flushed

    def release_pending(self) -> str:
        """Text held for a marker that never arrived; safe once the turn ends."""
        held, self.pending = self.pending, ""
        self.shown += held
        return held

    def release_suppressed(self) -> str:
        """Hand back the withheld markup; the caller decides what of it is shown."""
        held, self.suppressed_tail = self.suppressed_tail, ""
        self.markup_seen = False
        return held

    def _absorb_tool_calls(self, fragments: Sequence[Any]) -> list[str]:
        events: list[str] = []
        for position, fragment in enumerate(fragments):
            if not isinstance(fragment, dict):
                continue
            index = fragment.get("index")
            slot = index if isinstance(index, int) else position
            call = self.tool_calls.setdefault(
                slot,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            call_id = fragment.get("id")
            if isinstance(call_id, str) and call_id:
                call["id"] = call_id
            function = fragment.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                call["function"]["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] += arguments
            events.extend(self._provisional_events(call, arguments))
        return events

    def _provisional_events(self, call: dict[str, Any], fragment: Optional[str]) -> list[str]:
        """Open an early card for a large payload, then stream its argument text."""
        call_id = call["id"]
        name = call["function"]["name"]
        # a synthetic id cannot reconcile with the real tool_start, so it would strand a card.
        if not call_id or not name or name not in self._provisional_tool_names:
            return []
        events: list[str] = []
        arguments = call["function"]["arguments"]
        if call_id not in self.provisional_cards:
            if len(arguments) < PROVISIONAL_ARGS_MIN_CHARS:
                return []
            self.provisional_cards[call_id] = name
            events.append(
                _sse(
                    {
                        "type": "tool_start",
                        "tool_name": name,
                        "tool_call_id": call_id,
                        "arguments": {},
                        "provenance": tool_event_provenance(provisional = True),
                    }
                )
            )
        # first event carries the backlog the card missed, later ones just the fragment.
        if call_id not in self._args_streamed:
            self._args_streamed.add(call_id)
            text = arguments
        else:
            text = fragment or ""
        if text:
            events.append(
                _sse(
                    {
                        "type": "tool_args",
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "text": text,
                    }
                )
            )
        return events

    def ordered_tool_calls(self) -> list[dict[str, Any]]:
        return [self.tool_calls[key] for key in sorted(self.tool_calls)]


def _usage_chunk_line(model: str, totals: dict[str, int]) -> Optional[str]:
    """One summed usage chunk for the whole loop, or None when unreported.

    Per-turn usage chunks are withheld while the loop runs, so a multi-turn
    response reports the shape a single-turn one does instead of a burst of
    partial counts the client would have to add up itself.
    """
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


def _content_chunk_line(model: str, text: str) -> str:
    """A synthetic content delta, used to promote a reasoning-only answer."""
    return _sse(
        {
            "id": "chatcmpl-external-tools",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )


def _tool_names(tools: Sequence[Any]) -> list[str]:
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = (tool.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _provisional_card_names(
    tools: Sequence[Any],
    *,
    confirm_tool_calls: bool,
    bypass_permissions: bool,
    permission_mode: str,
) -> frozenset[str]:
    """Tools allowed an early card before their arguments finish streaming.

    Same gate as the GGUF loop: a call the user still has to approve shows no
    early card unless the card is only a text preview of the arguments, or the
    tool never prompts in auto mode.
    """
    from core.inference.tools import has_text_only_provisional_card, is_always_safe_tool

    allowed = set()
    for name in _tool_names(tools):
        confirm_gated = (
            confirm_tool_calls
            and not bypass_permissions
            and not (permission_mode == "auto" and is_always_safe_tool(name))
            and not has_text_only_provisional_card(name)
        )
        if not confirm_gated:
            allowed.add(name)
    return frozenset(allowed)


# returned instead of the generator's stopiteration, which cannot cross a thread boundary.
_STEP_DONE = object()


def _advance_tool_stream(generator: Any, outcome: dict[str, Any]) -> Any:
    try:
        return next(generator)
    except StopIteration as stop:
        outcome["result"] = stop.value
        return _STEP_DONE


def _close_provisional_cards(turn: _TurnAccumulator, resolved: set[str]) -> list[str]:
    """End every early card no execution claimed, so none spins forever."""
    lines = []
    for call_id, name in turn.provisional_cards.items():
        if call_id in resolved:
            continue
        resolved.add(call_id)
        lines.append(
            _sse(
                {
                    "type": "tool_end",
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "result": "",
                    "provenance": tool_event_provenance(provisional = True),
                }
            )
        )
    return lines


def _resolve_permission_mode(
    permission_mode: Optional[str], bypass_permissions: bool
) -> tuple[str, bool]:
    """Normalize the permission pair the way the local loops do."""
    if permission_mode == "full":
        return "full", True
    if bypass_permissions:
        return "full", True
    if permission_mode is None:
        return "auto", False
    if permission_mode not in ("ask", "auto", "off"):
        return "ask", False
    return permission_mode, False


async def stream_chat_completion_with_local_tools(
    client: Any,
    *,
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    tool_call_timeout: Optional[int] = None,
    auto_heal_tool_calls: bool = True,
    confirm_tool_calls: bool = False,
    bypass_permissions: bool = False,
    permission_mode: Optional[str] = None,
    disable_parallel_tool_use: bool = False,
    nudge_tool_calls: Optional[bool] = None,
    execute_tool: Optional[Callable[..., str]] = None,
    **stream_kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Stream a provider response, executing local tool calls in between.

    Yields OpenAI SSE ``data:`` lines: the provider's content chunks pass
    through, tool activity is inserted as the ``status`` / ``tool_start`` /
    ``tool_end`` events the frontend already renders, and one terminal chunk
    plus ``[DONE]`` close the stream.

    ``permission_mode`` mirrors the local loops: "ask" confirms every call,
    "auto" only those detected as high-risk, "off" never prompts (the sandbox
    stays on), "full" is the same switch as ``bypass_permissions``.
    """
    if execute_tool is None:
        from core.inference.tools import execute_tool as _default_execute_tool

        execute_tool = _default_execute_tool

    permission_mode, bypass_permissions = _resolve_permission_mode(
        permission_mode, bypass_permissions
    )

    conversation = list(messages)
    controller = ToolLoopController(tools = tools, auto_heal_tool_calls = auto_heal_tool_calls)
    all_tool_names = set(_tool_names(tools))
    cancel_event = threading.Event()
    usage_totals: dict[str, int] = {}
    # 9999+ is the local loops' "no limit" sentinel.
    effective_timeout = (
        None if tool_call_timeout is not None and tool_call_timeout >= 9999 else tool_call_timeout
    )
    remaining_iterations = max(1, max_tool_iterations)
    executed_calls = 0
    # budgeted apart from each other so a pre-tool nudge cannot spend the post-tool one.
    reprompt_count = 0
    post_tool_reprompts = 0
    last_reprompt_text = ""
    # only the opening turn resumes a partial; later turns start from a tool result.
    continue_final_message = bool(stream_kwargs.pop("continue_final_message", False))

    try:
        while True:
            # past the budget the catalog is dropped, so the last pass has to answer.
            tools_allowed = remaining_iterations > 0
            active_tools = controller.active_tools() if tools_allowed else []
            enabled_tool_names = set(_tool_names(active_tools))
            turn = _TurnAccumulator(
                provisional_tool_names = _provisional_card_names(
                    active_tools,
                    confirm_tool_calls = confirm_tool_calls,
                    bypass_permissions = bypass_permissions,
                    permission_mode = permission_mode,
                ),
            )
            gen = client.stream_chat_completion(
                messages = conversation,
                model = model,
                tools = active_tools or None,
                stream = True,
                continue_final_message = continue_final_message,
                **stream_kwargs,
            )
            continue_final_message = False
            try:
                async for line in gen:
                    for forwarded in turn.feed(line):
                        yield forwarded
            finally:
                try:
                    await gen.aclose()
                except RuntimeError:
                    # httpcore asyncgen cleanup on Python 3.13, suppressed as in the route.
                    pass
            _merge_usage(usage_totals, turn.usage)

            if turn.failed:
                # the error line already went out, so do not re-prompt from a half-finished turn.
                return

            # safety net for a model that writes its call as text: llama-server does
            # the same parse, so the call runs instead of printing itself.
            text_calls: list[dict[str, Any]] = []
            healed_calls: list[dict[str, Any]] = []
            if turn.markup_seen:
                healed_calls = parse_tool_calls_from_text(
                    turn.text,
                    allow_incomplete = True,
                    enabled_tool_names = all_tool_names,
                )[:MAX_TEXT_TOOL_CALLS_PER_TURN]
                # a turn that already carries structured calls runs those; the markup
                # it also wrote is display-only.
                if tools_allowed and not turn.wants_tools:
                    # healing decides what may run; display never depends on it.
                    text_calls = (
                        healed_calls
                        if auto_heal_tool_calls
                        else parse_tool_calls_from_text(
                            turn.text,
                            allow_incomplete = False,
                            enabled_tool_names = all_tool_names,
                        )[:MAX_TEXT_TOOL_CALLS_PER_TURN]
                    )
            held = turn.release_pending()
            if held:
                yield _content_chunk_line(model, held)
            if turn.suppressed_tail:
                # whatever survives the display strip was prose around the marker. a
                # marker that formed no call at all is restored whole, since stripping
                # an unclosed run to end-of-turn would eat the sentence with it.
                tail = turn.release_suppressed()
                visible = strip_tool_markup(
                    tail, final = True, enabled_tool_names = all_tool_names
                )
                if not visible.strip() and not healed_calls:
                    visible = tail
                if visible:
                    turn.shown += visible
                    yield _content_chunk_line(model, visible)
            pending_calls = turn.ordered_tool_calls() if turn.wants_tools else text_calls
            if disable_parallel_tool_use:
                pending_calls = pending_calls[:1]

            if not pending_calls or not tools_allowed:
                # tool_calls emitted after the catalog was dropped are ignored, never executed.
                for line in _close_provisional_cards(turn, set()):
                    yield line
                # a stalled turn describes the tool it means to use instead of calling
                # it. Nudge it once to act, as the local loops do. Gated on an empty
                # visible turn: OpenAI deltas are append-only, so text already streamed
                # cannot be taken back the way a cumulative snapshot can.
                intent_text = (turn.shown or turn.reasoning).strip()
                already_acted = any(record.executed for record in controller.history)
                used, cap = (
                    (post_tool_reprompts, MAX_POST_TOOL_REPROMPTS)
                    if already_acted
                    else (reprompt_count, MAX_ACT_REPROMPTS)
                )
                if (
                    tools_allowed
                    and auto_heal_tool_calls
                    # none keeps the default-on re-prompt; false disables it.
                    and (nudge_tool_calls is None or nudge_tool_calls)
                    and active_tools
                    and not turn.shown.strip()
                    and used < cap
                    and not is_reprompt_repeat(intent_text, last_reprompt_text)
                    and is_short_intent_without_action(intent_text)
                ):
                    reprompt_count += 1
                    if already_acted:
                        post_tool_reprompts += 1
                    last_reprompt_text = intent_text
                    logger.info(
                        "External local tool loop re-prompt %d/%d: model responded "
                        "without calling tools (%d chars)",
                        used + 1,
                        cap,
                        len(intent_text),
                    )
                    conversation.append({"role": "assistant", "content": intent_text})
                    tool_hint = " or ".join(_tool_names(active_tools)) or "an available tool"
                    conversation.append(
                        {"role": "user", "content": reprompt_to_act_message(tool_hint)}
                    )
                    # blank first so the badge clears, then name the pause so it is not a hang.
                    yield _status_sse("")
                    yield _status_sse(NUDGE_TOOL_CALLS_STATUS)
                    continue
                # whole reply arrived as reasoning (qwen3-style always-think models):
                # show it as the response rather than nothing. A length-truncated
                # thought is not an answer, so it is left in the thinking block.
                if (
                    not turn.shown.strip()
                    and turn.reasoning.strip()
                    and turn.finish_reason != "length"
                ):
                    yield _content_chunk_line(model, turn.reasoning)
                final_line = turn.final_chunk_line()
                if final_line is not None:
                    yield final_line
                break

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                # markup never replays: the call is carried structurally below.
                "content": strip_tool_markup(
                    turn.text, final = True, enabled_tool_names = enabled_tool_names
                ),
            }
            assistant_tool_calls: list[dict[str, Any]] = []
            tool_messages: list[dict[str, Any]] = []
            nudge_messages: list[dict[str, Any]] = []
            resolved_provisional: set[str] = set()

            for raw_call in pending_calls:
                decision = controller.prepare_call(raw_call)
                if not decision.should_execute:
                    completion = controller.record_noop(decision)
                    nudge_messages.append(completion.model_message())
                    resolved_provisional.add(decision.tool_call_id)
                    logger.info(
                        "Suppressed external local tool call as internal no-op: "
                        f"action={decision.action} tool={decision.tool_name}"
                    )
                    continue
                assistant_tool_calls.append(decision.as_assistant_tool_call())

                needs_confirm = (
                    bool(confirm_tool_calls)
                    and not bypass_permissions
                    and permission_mode != "off"
                )
                if needs_confirm and permission_mode == "auto":
                    from core.inference.tools import is_high_risk_tool_call

                    needs_confirm = is_high_risk_tool_call(decision.tool_name, decision.arguments)
                approval_id = new_approval_id() if needs_confirm else ""
                decision_slot = (
                    begin_tool_decision(session_id, approval_id) if needs_confirm else None
                )
                start_event = decision.tool_start_event()
                start_event["approval_id"] = approval_id
                start_event["awaiting_confirmation"] = needs_confirm

                try:
                    # a gated call has not started, so it must not read as running.
                    yield _status_sse(
                        awaiting_approval_status(decision.tool_name)
                        if needs_confirm
                        else decision.status_text
                    )
                    yield _sse(start_event)
                    verdict = None
                    if decision_slot is not None:
                        waiter = asyncio.ensure_future(
                            asyncio.to_thread(
                                wait_tool_decision,
                                decision_slot,
                                approval_id,
                                cancel_event,
                            )
                        )
                        try:
                            # delay the first keepalive so it is a separate write from the gated card.
                            done, _ = await asyncio.wait(
                                {waiter}, timeout = TOOL_APPROVAL_FLUSH_DELAY_S
                            )
                            while not done:
                                yield _SSE_KEEPALIVE
                                done, _ = await asyncio.wait(
                                    {waiter}, timeout = TOOL_HEARTBEAT_INTERVAL_S
                                )
                        finally:
                            if not waiter.done():
                                waiter.cancel()
                        verdict = waiter.result()
                    if verdict is not None and verdict != "deny":
                        yield _status_sse(decision.status_text)
                    if verdict == "deny":
                        decision_slot = None
                        resolved_provisional.add(decision.tool_call_id)
                        yield _sse(
                            {
                                "type": "tool_end",
                                "tool_name": decision.tool_name,
                                "tool_call_id": decision.tool_call_id,
                                "result": TOOL_REJECTED_MESSAGE,
                                "provenance": decision.provenance,
                            }
                        )
                        denied_message: dict[str, Any] = {
                            "role": "tool",
                            "name": decision.tool_name,
                            "content": TOOL_REJECTED_MESSAGE,
                        }
                        if decision.tool_call_id:
                            denied_message["tool_call_id"] = decision.tool_call_id
                        tool_messages.append(denied_message)
                        continue
                    decision_slot = None
                finally:
                    if decision_slot is not None:
                        abort_tool_decision(decision_slot, approval_id)

                def _invoke(output_callback: Callable[[str], None], call = decision) -> str:
                    kwargs: dict[str, Any] = {
                        "cancel_event": cancel_event,
                        "timeout": effective_timeout,
                        "session_id": session_id,
                        "thread_id": thread_id,
                        "disable_sandbox": bypass_permissions,
                    }
                    if accepts_output_callback(execute_tool):
                        kwargs["output_callback"] = output_callback
                    return execute_tool(call.tool_name, call.arguments, **kwargs)

                # same wrapper the local loops run tools through: live stdout for the
                # card, and a heartbeat so a long call cannot idle the stream out.
                tool_stream = stream_tool_execution(
                    _invoke,
                    tool_name = decision.tool_name,
                    tool_call_id = decision.tool_call_id,
                    cancel_event = cancel_event,
                )
                outcome: dict[str, Any] = {}
                try:
                    while True:
                        event = await asyncio.to_thread(
                            _advance_tool_stream, tool_stream, outcome
                        )
                        if event is _STEP_DONE:
                            break
                        if event.get("type") == "heartbeat":
                            yield _SSE_KEEPALIVE
                        else:
                            yield _sse(event)
                    result = outcome.get("result", "")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reported back to the model
                    logger.exception("External local tool %s raised: %s", decision.tool_name, exc)
                    result = f"Error: tool raised an exception: {exc}"
                finally:
                    tool_stream.close()
                completion = controller.record_result(decision, result)
                executed_calls += 1
                resolved_provisional.add(decision.tool_call_id)
                yield _sse(completion.tool_end_event())
                tool_messages.append(completion.tool_message())

            for line in _close_provisional_cards(turn, resolved_provisional):
                yield line
            # an empty status clears the UI badge between iterations.
            yield _status_sse("")
            if assistant_tool_calls:
                assistant_message["tool_calls"] = assistant_tool_calls
            if assistant_message["content"] or assistant_tool_calls:
                conversation.append(assistant_message)
            conversation.extend(tool_messages)
            append_deferred_nudges(conversation, nudge_messages)
            remaining_iterations -= 1
            if remaining_iterations == 0 and not controller.force_final_answer:
                # the catalog is gone from here on, so say why rather than letting
                # the next pass ask for a tool that is no longer offered.
                conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})

        usage_line = _usage_chunk_line(model, usage_totals)
        if usage_line is not None:
            yield usage_line
        yield "data: [DONE]"
        logger.info(
            "External local tool loop finished (model=%s, executed_tool_calls=%d)",
            model,
            executed_calls,
        )
    finally:
        # a disconnect closes this generator; release any tool still blocking in its thread.
        cancel_event.set()
