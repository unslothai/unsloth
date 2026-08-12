# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Agentic tool loop for external OpenAI-compatible providers.

Self-hosted endpoints (llama.cpp / vLLM / Ollama / a generic custom server)
speak the OpenAI function-calling contract but have no server-side tools of
their own, so the provider-side ``web_search`` / ``code_execution`` builtins
never reach them. This module runs Unsloth's local tool loop against such an
endpoint:

* Each turn streams a chat completion carrying the active tool schemas and
  accumulates ``tool_calls`` (structured deltas, or text ``<tool_call>``
  markup when auto-heal is on).
* Executes each call with the local tool executor (``web_search`` / ``python``
  / ``terminal``), streams ``tool_output`` / ``heartbeat`` events, and feeds
  the results back as ``role=tool`` messages.
* Re-enters the model until a final answer, bounded by
  ``max_tool_iterations``.

The loop yields the same flat event dicts as the GGUF / safetensors local
loops (``status`` / ``content`` / ``tool_start`` / ``tool_end`` /
``tool_output`` / ``heartbeat``), so the existing route converters and
frontend tool cards work unchanged.

This is an independent implementation that fixes the bug reported in
unslothai/unsloth#7282 ("Remote Models not able to do Tool Calling"): the
Search / Code / MCP toggles stayed greyed out for remote Ollama / llama.cpp /
vLLM / Custom connections because they were treated as "no local tools". A
similar fix was proposed earlier in unslothai/unsloth#7330; that PR and this
code were written separately. This implementation differs from #7330 in that
the self-hosted capability is declared in the backend ``PROVIDER_REGISTRY``
(the provider-level ``studio_tools`` wildcard, see
``core.inference.providers.provider_runs_local_tools``) instead of a
hardcoded provider-type list, so it rides the same registry-driven capability
mechanism upstream uses for Codex (``providerModelSupportsStudioTools`` /
``supportsStudioToolsForThisTurn``) and supersedes #7330.
"""

import asyncio
import json
import threading
from typing import Any, AsyncGenerator, Optional

import structlog

from core.inference.external_provider import (
    ExternalProviderClient,
)
from core.inference.providers import PROVIDER_REGISTRY
from core.inference.tool_call_parser import (
    BUDGET_EXHAUSTED_NUDGE,
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from core.inference.tool_loop_controller import (
    ToolLoopController,
    append_deferred_nudges,
    awaiting_approval_status,
)
from core.inference.tool_stream_exec import (
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

logger = structlog.get_logger(__name__)

# Provider types that run Unsloth's LOCAL tool loop. Every one is a
# user-supplied OpenAI-compatible base URL (llama.cpp / vLLM / Ollama / a
# generic custom server) with no server-side tools, so Unsloth may legitimately
# execute the tools on its own machine. Derived from the registry's
# provider-level ``studio_tools`` capability (the ``"*"`` wildcard in
# ``model_capabilities``, see ``core.inference.providers``), which is the
# single source of truth for "Studio executes tools for this provider"; the
# frontend reads the same flag via ``providerModelSupportsStudioTools``, so the
# gate here and the UI pill can never drift apart.
EXTERNAL_LOCAL_TOOL_PROVIDERS = frozenset(
    provider_type
    for provider_type, info in PROVIDER_REGISTRY.items()
    if (info.get("model_capabilities") or {}).get("*", {}).get("studio_tools")
)


# Cap the tool calls taken from a single round, mirroring ``_MAX_TOOL_CALLS_PER_TURN``
# in the local llama.cpp and safetensors loops. llama-server constrains its own fan-out
# with a grammar; a remote OAI-compat server does not, so one turn could otherwise ask for
# dozens of parallel calls and have them all scheduled before the budget check.
_MAX_TOOL_CALLS_PER_TURN = 8


def _delta_text(content: Any) -> str:
    """Text from a streamed content delta.

    Some OAI-compat servers send structured parts (a list of ``{type: "text",
    text: ...}`` objects) instead of a plain string; ``str()`` on that list would
    leak a Python repr into the answer and into the next round's transcript.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            part_text = part.get("text")
            if part_type in ("text", "input_text") and isinstance(part_text, str):
                text_parts.append(part_text)
        return "".join(text_parts)
    return "" if content is None else str(content)


def _active_tool_names(tools: list[dict]) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    return names


async def _consume_tool_execution(
    gen: Any,
    *,
    tool_name: str,
    cancel_event: Any = None,
) -> AsyncGenerator[dict, None]:
    """Bridge the sync ``stream_tool_execution`` generator into this async loop.

    The sync generator runs in a daemon worker thread; its event dicts
    (``tool_output`` / ``heartbeat``) are forwarded as yielded events and the
    final result string is emitted as a trailing ``{"type": "_result", ...}``
    event.

    Abnormal exit (the consumer closes this generator early, e.g. an SSE
    disconnect) sets ``cancel_event`` so a cancel-observing tool stops instead
    of running to completion on the user's machine after the client has left,
    then joins the worker with a bounded timeout off the event loop. ``gen`` is
    deliberately NOT closed from here: it is being driven by the worker thread,
    and closing it would raise "generator already executing"; the tool stops
    through ``cancel_event`` instead.
    """
    queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _put(kind: str, payload: Any) -> None:
        # call_soon_threadsafe: an asyncio.Queue put_nowait from a worker
        # thread does NOT wake the event-loop selector, so a slow tool would
        # leave the loop parked forever waiting on get(). Routing through the
        # loop is the thread-safe wake-up path.
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def _worker() -> None:
        try:
            while True:
                try:
                    event = next(gen)
                except StopIteration as stop:
                    _put("_result", stop.value)
                    return
                except BaseException as exc:  # noqa: BLE001 - re-raised on the caller side
                    _put("_error", exc)
                    return
                _put("_event", event)
        except BaseException as exc:  # noqa: BLE001
            _put("_error", exc)

    thread = threading.Thread(
        target = _worker,
        daemon = True,
        name = f"external-tool-exec-{tool_name or 'unknown'}",
    )
    thread.start()
    error: Any = None
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "_event":
                yield payload
            elif kind == "_error":
                # A tool exception is a HANDLED path: captured here, raised
                # after the finally (like stream_tool_execution does), so the
                # loop's ``except`` can feed it back as a tool result. It must
                # never set the shared cancel_event.
                error = payload
                break
            else:
                # Final tool result, delivered as a trailing event so the loop
                # can capture it from the same ``async for``.
                yield {"type": "_result", "result": payload}
                break
    except BaseException:
        # Consumer abandoned this generator (GeneratorExit via ``aclose()``, or
        # CancelledError from the driving task): stop a running tool so it
        # doesn't execute to completion after the client left. A tool error
        # never takes this path -- it is captured as ``error`` above.
        if cancel_event is not None:
            cancel_event.set()
        raise
    finally:
        # The worker has already exited on a clean finish (it returns after the
        # StopIteration above), so this join returns immediately there. On an
        # abnormal exit it waits for the aborting tool; bounded so teardown
        # never blocks the event loop (moved to a thread).
        await asyncio.to_thread(thread.join, 5)
    if error is not None:
        raise error


async def run_external_provider_tool_loop(
    *,
    client: ExternalProviderClient,
    messages: list[dict],
    model: str,
    tools: list[dict],
    execute_tool: Any,
    cancel_event: Any = None,
    auto_heal_tool_calls: bool = True,
    max_tool_iterations: int = 25,
    tool_call_timeout: int = 300,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    rag_scope: Optional[dict] = None,
    bypass_permissions: bool = False,
    permission_mode: Optional[str] = None,
    confirm_tool_calls: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    top_k: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Drive a local-execution tool loop against an external OAI-compatible endpoint.

    ``client`` must be an ``ExternalProviderClient`` pointing at a provider in
    ``EXTERNAL_LOCAL_TOOL_PROVIDERS``. ``execute_tool(name, arguments, ...)``
    is Unsloth's local tool executor (``web_search`` / ``python`` /
    ``terminal``), which may accept ``output_callback`` / ``cancel_event`` /
    ``session_id`` / ``thread_id`` / ``rag_scope`` / ``disable_sandbox``.

    Yields flat local-loop events; the caller converts them to SSE.
    """
    conversation = list(messages)
    unrestricted_tools = not tools
    # Same permission normalization as the GGUF / safetensors loops: "full"
    # and bypass_permissions are the same switch; unset defaults to "auto".
    if permission_mode == "full":
        bypass_permissions = True
    elif permission_mode is None:
        permission_mode = "auto"

    tool_controller = ToolLoopController(
        tools = (None if unrestricted_tools else tools),
        auto_heal_tool_calls = auto_heal_tool_calls,
    )
    # Names gate the text-tool-call fallback (bare rehearsal / wrapper-less
    # forms) on the active catalog, so an ordinary JSON answer isn't misread.
    _enabled_names_gate = None if unrestricted_tools else set(_active_tool_names(tools))

    if max_tool_iterations <= 0:
        # 0 = disabled (same contract as the GGUF loop).
        yield {"type": "status", "text": ""}
        return

    final_attempt_done = False
    executed_tool_iters = 0

    for iteration in range(max_tool_iterations + 1):
        if cancel_event is not None and cancel_event.is_set():
            return
        yield {"type": "status", "text": ""}

        if final_attempt_done:
            active_tools: list[dict] = []
        else:
            active_tools = tool_controller.active_tools()
            if not active_tools and not unrestricted_tools:
                final_attempt_done = True
                active_tools = []

        # ---- Single turn against the endpoint ----
        content = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        gen = client.stream_chat_completion(
            messages = conversation,
            model = model,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            presence_penalty = presence_penalty,
            top_k = top_k,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            enabled_tools = None,
            tools = (active_tools or None),
            tool_choice = None,
            stream = True,
        )
        try:
            async for line in gen:
                stripped = line.strip()
                if not stripped.startswith("data:"):
                    continue
                payload = stripped[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    message = (
                        chunk["error"].get("message") if isinstance(chunk["error"], dict) else ""
                    ) or str(chunk["error"])
                    raise RuntimeError(f"External provider error: {message}")
                for choice in chunk.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") or {}
                    if isinstance(delta, dict):
                        delta_text = _delta_text(delta.get("content"))
                        if delta_text:
                            content += delta_text
                            yield {"type": "content", "text": content}
                        for tc in delta.get("tool_calls") or []:
                            if not isinstance(tc, dict):
                                continue
                            try:
                                idx = int(tc.get("index", 0))
                            except (TypeError, ValueError):
                                idx = 0
                            slot = tool_calls_acc.setdefault(
                                idx,
                                {
                                    "id": tc.get("id") or f"call_{idx}",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if not isinstance(fn, dict):
                                continue
                            if fn.get("name"):
                                slot["function"]["name"] = fn["name"]
                            args_fragment = fn.get("arguments")
                            if isinstance(args_fragment, str) and args_fragment:
                                slot["function"]["arguments"] += args_fragment
        finally:
            try:
                await gen.aclose()
            except RuntimeError:
                pass  # suppress httpcore asyncgen cleanup error (Python 3.13 + httpcore 1.0.x)

        calls = list(tool_calls_acc.values())
        # Structured deltas won this turn; otherwise fall back to healing tool
        # markup written as plain text (llama.cpp models with weaker templates).
        text_parsed = False
        if not calls and auto_heal_tool_calls and has_tool_signal(content):
            text_calls = parse_tool_calls_from_text(
                content,
                allow_incomplete = True,
                enabled_tool_names = _enabled_names_gate,
            )
            if text_calls:
                calls = text_calls
                text_parsed = True

        # Collapse exact-duplicate calls and cap the fan-out for one turn,
        # mirroring the local GGUF / safetensors loops. A remote OAI-compat
        # server does not constrain its own parallelism, so a single turn could
        # otherwise schedule dozens of executions past the iteration budget.
        if len(calls) > 1:
            _seen_keys: set = set()
            _deduped: list = []
            for _tc in calls:
                _fn = _tc.get("function") or {}
                _key = (_fn.get("name", ""), str(_fn.get("arguments", "")))
                if _key in _seen_keys:
                    continue
                _seen_keys.add(_key)
                _deduped.append(_tc)
                if len(_deduped) >= _MAX_TOOL_CALLS_PER_TURN:
                    break
            if len(_deduped) != len(calls):
                logger.warning(
                    "external_provider.local_tools.turn_call_cap",
                    total = len(calls),
                    kept = len(_deduped),
                )
            calls = _deduped

        if not calls:
            # Final answer (or the endpoint never emits tool calls): done.
            break

        assistant_content = content
        if text_parsed:
            assistant_content = strip_tool_markup(
                content,
                final = True,
                enabled_tool_names = _enabled_names_gate,
            )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": list(calls),
            }
        )

        turn_executed = False
        deferred_noop_msgs: list[Any] = []
        for tc in calls:
            decision = tool_controller.prepare_call(tc)
            if not decision.should_execute:
                deferred_noop_msgs.append(tool_controller.record_noop(decision).model_message())
                continue

            turn_executed = True

            # Interactive confirm gate, mirroring the local GGUF / safetensors
            # loops: with confirm_tool_calls the loop parks on the shared
            # approval slot before a call starts, unless bypass_permissions or
            # permission_mode "off" opts out. "auto" pauses only high-risk calls.
            needs_confirm = (
                bool(confirm_tool_calls) and not bypass_permissions and permission_mode != "off"
            )
            if needs_confirm and permission_mode == "auto":
                from core.inference.tools import is_high_risk_tool_call
                needs_confirm = is_high_risk_tool_call(decision.tool_name, decision.arguments)
            approval_id = new_approval_id() if needs_confirm else ""
            decision_slot = begin_tool_decision(session_id, approval_id) if needs_confirm else None
            start_event = decision.tool_start_event()
            start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirm

            try:
                yield {
                    "type": "status",
                    "text": (
                        awaiting_approval_status(decision.tool_name)
                        if needs_confirm
                        else decision.status_text
                    ),
                }
                yield start_event

                # The approval prompt can sit unanswered for a while; emit SSE
                # keepalives while the user decides so a proxy / the WebView
                # doesn't drop the stream (the GGUF loop has no such transport,
                # but this one streams over the external SSE path).
                _decision = None
                if decision_slot is not None:
                    _wait_task = asyncio.ensure_future(
                        asyncio.to_thread(
                            wait_tool_decision,
                            decision_slot,
                            approval_id,
                            cancel_event = cancel_event,
                        )
                    )
                    while not _wait_task.done():
                        yield {"type": "heartbeat"}
                        await asyncio.sleep(0.5)
                    _decision = _wait_task.result()
                if _decision == "deny":
                    decision_slot = None
                    yield {
                        "type": "tool_end",
                        "tool_name": decision.tool_name,
                        "tool_call_id": decision.tool_call_id,
                        "result": TOOL_REJECTED_MESSAGE,
                        "provenance": decision.provenance,
                    }
                    denied_message = {
                        "role": "tool",
                        "name": decision.tool_name,
                        "content": TOOL_REJECTED_MESSAGE,
                    }
                    if decision.tool_call_id:
                        denied_message["tool_call_id"] = decision.tool_call_id
                    conversation.append(denied_message)
                    continue
                decision_slot = None
                if needs_confirm:
                    # Approved: now it really is running.
                    yield {"type": "status", "text": decision.status_text}
            finally:
                if decision_slot is not None:
                    abort_tool_decision(decision_slot, approval_id)

            eff_timeout = None if tool_call_timeout >= 9999 else tool_call_timeout

            def _invoke_tool(output_callback: Any, _decision = decision) -> str:
                kwargs: dict[str, Any] = dict(
                    cancel_event = cancel_event,
                    timeout = eff_timeout,
                    session_id = session_id,
                    thread_id = thread_id,
                    rag_scope = rag_scope,
                    disable_sandbox = bypass_permissions,
                )
                if accepts_output_callback(execute_tool):
                    kwargs["output_callback"] = output_callback
                return execute_tool(_decision.tool_name, _decision.arguments, **kwargs)

            result = None
            try:
                async for ev in _consume_tool_execution(
                    stream_tool_execution(
                        _invoke_tool,
                        tool_name = decision.tool_name,
                        tool_call_id = decision.tool_call_id,
                        cancel_event = cancel_event,
                    ),
                    tool_name = decision.tool_name,
                    cancel_event = cancel_event,
                ):
                    if ev.get("type") == "_result":
                        result = ev.get("result")
                    else:
                        yield ev
            except Exception as exc:  # noqa: BLE001 - surfaced as a tool result
                logger.warning("External tool %s raised: %s", decision.tool_name, exc)
                result = f"Error: tool raised an exception: {exc}"

            completion = tool_controller.record_result(decision, result)
            yield completion.tool_end_event()
            conversation.append(completion.tool_message())

        append_deferred_nudges(conversation, deferred_noop_msgs)
        yield {"type": "status", "text": ""}

        if tool_controller.force_final_answer:
            final_attempt_done = True
            continue
        if not unrestricted_tools and not tool_controller.active_tools():
            final_attempt_done = True
            continue
        if turn_executed:
            executed_tool_iters += 1
        if executed_tool_iters >= max_tool_iterations and not final_attempt_done:
            # Budget exhausted; nudge a final plain answer.
            final_attempt_done = True
            conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})

    yield {"type": "status", "text": ""}
