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
catalog would duplicate or conflict with those. Selected RAG and MCP tools may
join the self-hosted catalog because Studio still owns their execution.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from typing import Any, AsyncGenerator, Callable, Optional, Sequence

from loggers import get_logger

from core.inference.chat_template_helpers import (
    append_assistant_turn,
    neutralize_tool_descriptions,
    reconciled_tool_choice,
)
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
    if not provider_type:
        return False
    from core.inference.providers import provider_runs_local_tools

    return provider_runs_local_tools(provider_type)


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


def _delta_text(content: Any) -> str:
    """Normalize text from string or structured OpenAI-compatible deltas."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


# nested count blocks a provider may report alongside the headline totals.
_USAGE_DETAIL_FIELDS = (
    "prompt_tokens_details",
    "completion_tokens_details",
    "input_tokens_details",
    "output_tokens_details",
)


def _merge_usage(totals: dict[str, Any], usage: Any) -> None:
    """Sum every countable field of one turn's usage block into ``totals``.

    Details are summed alongside the headline totals: the frontend reads
    ``prompt_tokens_details.cached_tokens`` for its cache readout and pricing reads
    the reasoning slice, so keeping only the totals would make a tool-loop response
    report less than the same provider's plain stream.
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
        single_call_only: bool = False,
    ) -> None:
        # only the first call survives disable_parallel_tool_use, so the rest must open
        # no card: it would stream its arguments and then close blank, having never run.
        self._single_call_only = single_call_only
        self.text: str = ""
        self.shown: str = ""
        self.pending: str = ""
        self.suppressed_tail: str = ""
        self.markup_seen: bool = False
        self.reasoning: str = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: Optional[str] = None
        self.failed: bool = False
        self.usage: dict[str, Any] = {}
        self.provisional_cards: dict[str, str] = {}
        self._provisional_tool_names = provisional_tool_names
        self._real_id_slots: set[int] = set()
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
            # an sse comment is the provider's keepalive; dropping it idles the stream out.
            return [line] if line.startswith(":") else []
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
        content = _delta_text(delta.get("content"))
        if content:
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
            # a server that omits the id still needs one: the replayed call and its tool
            # message are paired by it, as in llama_cpp.py's accumulator.
            fallback_id = f"call_{slot}"
            call = self.tool_calls.setdefault(
                slot,
                {"id": fallback_id, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            call_id = fragment.get("id")
            if isinstance(call_id, str) and call_id:
                call["id"] = call_id
                self._real_id_slots.add(slot)
            function = fragment.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                # replaced, never appended: llama-server re-sends the whole name as it
                # grows mid-parse (common/chat.cpp compute_diffs), so += would double it.
                call["function"]["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] += arguments
            if self._single_call_only and slot != min(self.tool_calls):
                continue
            events.extend(
                self._provisional_events(call, arguments, has_real_id = slot in self._real_id_slots)
            )
        return events

    def _provisional_events(
        self, call: dict[str, Any], fragment: Optional[str], *, has_real_id: bool
    ) -> list[str]:
        """Open an early card for a large payload, then stream its argument text."""
        call_id = call["id"]
        name = call["function"]["name"]
        # a synthetic id may still be replaced by a real one, which would strand the card.
        if not has_real_id or not name or name not in self._provisional_tool_names:
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


def _usage_chunk_line(model: str, totals: dict[str, Any]) -> Optional[str]:
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


async def _drain_step_task(task: Optional["asyncio.Future"], cancel_event: threading.Event) -> None:
    """Wait for a pending ``next(gen)`` worker before its generator is closed.

    Cancelling the awaiting task does not stop the worker thread, and calling
    ``close()`` while ``next()`` is still running raises ``ValueError: generator
    already executing`` and skips the generator's own cleanup. Setting the cancel
    flag lets a cancel-observing tool return, then the task is shielded until the
    worker actually finishes.
    """
    if task is None:
        return
    if task.done():
        # its worker already returned, and the loop recovers from a tool error below,
        # so signalling cancellation here would abort the rest of the request.
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
    if task.done():
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass


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
    rag_scope: Any = None,
    tool_choice: Any = None,
    execute_tool: Optional[Callable[..., str]] = None,
    cancel_event: Optional[threading.Event] = None,
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
    # only the opening turn resumes a partial; later turns start from a tool result.
    continue_final_message = bool(stream_kwargs.pop("continue_final_message", False))

    # first-pass retrieval as in the other loops: the nudge promises passages, so send them.
    skip_autoinject = continue_final_message or (
        confirm_tool_calls and not bypass_permissions and permission_mode not in ("auto", "off")
    )
    autoinject = None
    if not skip_autoinject:
        from core.inference.tools import build_rag_autoinject
        autoinject = await asyncio.to_thread(build_rag_autoinject, conversation, rag_scope)
    if autoinject:
        for event in autoinject["events"]:
            yield _sse(event)
        conversation.extend(autoinject["messages"])
    # a KB search ran outside the controller, so the loop opens in its post-tool phase.
    rag_autoinjected = bool(autoinject)

    # the client drops a markup-carrying tool from the catalog it sends, so authorize
    # against the same sanitized list or a withheld tool could still be executed (#7066).
    requested_tools = tools
    tools = neutralize_tool_descriptions(tools)
    controller = ToolLoopController(tools = tools, auto_heal_tool_calls = auto_heal_tool_calls)
    all_tool_names = set(_tool_names(tools))
    # the route owns it when Stop has to reach a running tool through /inference/cancel.
    if cancel_event is None:
        cancel_event = threading.Event()
    usage_totals: dict[str, Any] = {}
    # 9999+ is the local loops' "no limit" sentinel.
    effective_timeout = (
        None if tool_call_timeout is not None and tool_call_timeout >= 9999 else tool_call_timeout
    )
    remaining_iterations = max(0, max_tool_iterations)
    executed_calls = 0
    # budgeted apart from each other so a pre-tool nudge cannot spend the post-tool one.
    reprompt_count = 0
    post_tool_reprompts = 0
    last_reprompt_text = ""
    # only the request is one-shot; the flag itself still merges the first generated turn.
    resume_partial_request = continue_final_message
    # the client sees only the sanitized list, so a choice forcing a dropped tool has
    # to be downgraded here or the request names a function absent from `tools`.
    forced_tool_choice = reconciled_tool_choice(tool_choice, requested_tools, tools)
    tool_denied = False

    try:
        while True:
            # Stop arrives as a POST that sets this event, so every boundary honours it.
            if cancel_event.is_set():
                break
            # past the budget the catalog is dropped, so the last pass has to answer.
            tools_allowed = remaining_iterations > 0 and not controller.force_final_answer
            active_tools = controller.active_tools() if tools_allowed else []
            enabled_tool_names = set(_tool_names(active_tools))
            turn = _TurnAccumulator(
                provisional_tool_names = _provisional_card_names(
                    active_tools,
                    confirm_tool_calls = confirm_tool_calls,
                    bypass_permissions = bypass_permissions,
                    permission_mode = permission_mode,
                ),
                single_call_only = disable_parallel_tool_use,
            )
            gen = client.stream_chat_completion(
                messages = conversation,
                model = model,
                tools = active_tools or None,
                tool_choice = forced_tool_choice if active_tools else None,
                stream = True,
                continue_final_message = resume_partial_request,
                **stream_kwargs,
            )
            resume_partial_request = False
            try:
                async for line in gen:
                    if cancel_event.is_set():
                        break
                    for forwarded in turn.feed(line):
                        yield forwarded
            finally:
                try:
                    await gen.aclose()
                except RuntimeError:
                    # httpcore asyncgen cleanup on Python 3.13, suppressed as in the route.
                    pass
            _merge_usage(usage_totals, turn.usage)

            if cancel_event.is_set():
                for line in _close_provisional_cards(turn, set()):
                    yield line
                break

            if turn.failed:
                # the error line already went out, so do not re-prompt from a half-finished turn.
                # any early card still has to close, or the frontend keeps it spinning.
                for line in _close_provisional_cards(turn, set()):
                    yield line
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
                visible = strip_tool_markup(tail, final = True, enabled_tool_names = all_tool_names)
                if not visible.strip() and not healed_calls:
                    visible = tail
                if visible:
                    turn.shown += visible
                    yield _content_chunk_line(model, visible)
            pending_calls = turn.ordered_tool_calls() if turn.wants_tools else text_calls
            if disable_parallel_tool_use:
                pending_calls = pending_calls[:1]
            if pending_calls and tools_allowed:
                forced_tool_choice = None

            if not pending_calls or not tools_allowed:
                # tool_calls emitted after the catalog was dropped are ignored, never executed.
                for line in _close_provisional_cards(turn, set()):
                    yield line
                # a stalled turn describes the tool it means to use instead of calling
                # it. Nudge it once to act, as the local loops do. Gated on an empty
                # visible turn: OpenAI deltas are append-only, so text already streamed
                # cannot be taken back the way a cumulative snapshot can.
                intent_text = (turn.shown or turn.reasoning).strip()
                already_acted = rag_autoinjected or any(
                    record.executed for record in controller.history
                )
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
                    and not tool_denied
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
                    # merges into a resumed partial; the nudge below is a user turn.
                    append_assistant_turn(
                        conversation,
                        {"role": "assistant", "content": intent_text},
                        continue_final_message = continue_final_message,
                    )
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
                # the full catalog, so a spent one-shot's rehearsal form still strips.
                "content": strip_tool_markup(
                    turn.text, final = True, enabled_tool_names = all_tool_names
                ),
            }
            assistant_tool_calls: list[dict[str, Any]] = []
            tool_messages: list[dict[str, Any]] = []
            nudge_messages: list[dict[str, Any]] = []
            resolved_provisional: set[str] = set()
            turn_executed_real_tool = False

            for raw_call in pending_calls:
                if cancel_event.is_set():
                    break
                decision = controller.prepare_call(raw_call)
                if not decision.should_execute:
                    completion = controller.record_noop(decision)
                    nudge_messages.append(completion.model_message())
                    # left unresolved on purpose: _close_provisional_cards ends any early
                    # card this id opened, which no execution will ever close.
                    logger.info(
                        "Suppressed external local tool call as internal no-op: "
                        f"action={decision.action} tool={decision.tool_name}"
                    )
                    continue
                assistant_tool_calls.append(decision.as_assistant_tool_call())

                needs_confirm = (
                    bool(confirm_tool_calls) and not bypass_permissions and permission_mode != "off"
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
                        tool_denied = True
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
                        "rag_scope": rag_scope,
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
                step_task: Optional[asyncio.Future] = None
                try:
                    while True:
                        step_task = asyncio.create_task(
                            asyncio.to_thread(_advance_tool_stream, tool_stream, outcome)
                        )
                        # wait leaves the worker future pending when this coroutine is
                        # cancelled, so the drain below can still join it; await would
                        # cancel the future and let close() race the running next().
                        await asyncio.wait({step_task})
                        event = step_task.result()
                        step_task = None
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
                    await _drain_step_task(step_task, cancel_event)
                    tool_stream.close()
                completion = controller.record_result(decision, result)
                executed_calls += 1
                turn_executed_real_tool = True
                # opens the post-tool phase; a carried-over stall text would eat its nudge.
                last_reprompt_text = ""
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
                # merges into a resumed partial, so a continued tool turn stays one message.
                append_assistant_turn(
                    conversation,
                    assistant_message,
                    continue_final_message = continue_final_message,
                )
            conversation.extend(tool_messages)
            # deferred to after the tool results, so a no-op never splits a call from them.
            _append_user_turn(
                conversation,
                "\n\n".join(dict.fromkeys(message["content"] for message in nudge_messages)),
            )
            if turn_executed_real_tool:
                remaining_iterations -= 1
            if remaining_iterations == 0 and not controller.force_final_answer:
                # the catalog is gone from here on, so say why rather than letting
                # the next pass ask for a tool that is no longer offered.
                _append_user_turn(conversation, BUDGET_EXHAUSTED_NUDGE)

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
