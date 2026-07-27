# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local tool loop for OpenAI-compatible external Connections (#7282).

Ollama / llama.cpp / vLLM / Custom Connections speak `/v1/chat/completions` on
a remote host, but Unsloth's Search / Code / MCP tools still execute on the
Studio machine. This module drives that loop:

1. Stream a chat completion with OpenAI function tools attached.
2. Forward content deltas as OpenAI SSE chunks.
3. On ``tool_calls``, confirm (optional) + execute locally, emit
   ``tool_start`` / ``tool_end`` events the chat UI already understands.
4. Append the assistant tool-call message + tool results and continue.

Hosted providers (OpenAI / Anthropic / …) stay on the pure-proxy +
``supportsBuiltin*`` path — they must not enter this loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from core.inference.tool_loop_controller import (
    ToolLoopController,
    strip_result_for_model,
    tool_event_provenance,
)
from core.inference.tools import (
    execute_tool,
    is_always_safe_tool,
    is_potentially_unsafe_tool_call,
)
from state.tool_approvals import (
    TOOL_REJECTED_MESSAGE,
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)

logger = logging.getLogger(__name__)

# OAI-compat Connections that can drive Unsloth's local tool runtime.
LOCAL_TOOL_RUNTIME_PROVIDER_TYPES = frozenset({"ollama", "llama_cpp", "vllm", "custom"})

# Cap the tool calls taken from a single round, mirroring `_MAX_TOOL_CALLS_PER_TURN` in
# the local llama.cpp and safetensors loops. llama-server constrains its own fan-out with
# a grammar; a remote OAI-compat server does not, so one turn could otherwise ask for
# hundreds of parallel calls and have them all scheduled before the budget check.
_MAX_TOOL_CALLS_PER_TURN = 8

# Comment line that keeps the SSE connection warm while a tool blocks. Same payload as
# `_OPENAI_PASSTHROUGH_SSE_KEEPALIVE` in routes/inference.py; the caller appends the blank
# line that terminates the event.
_SSE_KEEPALIVE_LINE = ": keep-alive"
_TOOL_STREAM_KEEPALIVE_S = 15.0


def provider_supports_local_tool_runtime(provider_type: Optional[str]) -> bool:
    """True for OAI-compat Connections that may use Studio's local tools."""
    return (provider_type or "") in LOCAL_TOOL_RUNTIME_PROVIDER_TYPES


async def _run_with_keepalive(coro: Any, out: list[Any]) -> AsyncGenerator[str, None]:
    """Yield SSE keepalive comments while ``coro`` runs, then append its result to ``out``.

    A terminal / python / MCP tool, or a human deciding on an approval prompt, can hold the
    loop for minutes between ``tool_start`` and ``tool_end``. With no bytes on the wire in
    that window an intermediate proxy or the browser drops the stream, and the user loses
    the answer the tool was fetching. Exceptions from ``coro`` surface on the final
    iteration so callers keep their existing try/except semantics.
    """
    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout = _TOOL_STREAM_KEEPALIVE_S)
            if done:
                break
            yield _SSE_KEEPALIVE_LINE
    finally:
        if not task.done():
            task.cancel()
    out.append(task.result())


def _parse_sse_data_line(line: str) -> Optional[dict[str, Any]]:
    """Parse one ``data: {...}`` SSE line into a dict; ignore heartbeats/[DONE]."""
    if not line:
        return None
    text = line.strip()
    if text.startswith("data:"):
        text = text[5:].strip()
    if not text or text == "[DONE]":
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _merge_tool_call_delta(acc: dict[int, dict[str, Any]], tc_delta: dict[str, Any]) -> None:
    """Accumulate a streaming ``delta.tool_calls[]`` fragment into ``acc``."""
    try:
        idx = int(tc_delta.get("index", 0))
    except (TypeError, ValueError):
        idx = 0
    slot = acc.setdefault(
        idx,
        {
            "id": "",
            "type": "tool",
            "function": {"name": "", "arguments": ""},
        },
    )
    if tc_delta.get("id"):
        slot["id"] = str(tc_delta["id"])
    if tc_delta.get("type"):
        slot["type"] = str(tc_delta["type"])
    func = tc_delta.get("function") or {}
    if isinstance(func, dict):
        if func.get("name"):
            slot["function"]["name"] += str(func["name"])
        if func.get("arguments"):
            slot["function"]["arguments"] += str(func["arguments"])


def _coerce_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse tool-call argument JSON; fall back to ``{}`` on malformed input."""
    if not raw or not str(raw).strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _openai_content_chunk_line(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
    finish_reason: Optional[str] = None,
) -> str:
    """Build one OpenAI chat-completion SSE ``data:`` line for a content delta."""
    choice: dict[str, Any] = {
        "index": 0,
        "delta": {"content": content} if content else {},
        "finish_reason": finish_reason,
    }
    if not content and finish_reason is None:
        choice["delta"] = {}
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload, ensure_ascii = False)}"


async def stream_external_local_tool_loop(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    top_k: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    tool_choice: Optional[Any] = None,
    confirm_tool_calls: bool = False,
    bypass_permissions: bool = False,
    permission_mode: Optional[str] = None,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    tool_call_timeout: int = 300,
    max_tool_iterations: int = 25,
    rag_scope: Optional[dict] = None,
    cancel_event: Optional[threading.Event] = None,
    completion_id: Optional[str] = None,
    auto_heal_tool_calls: bool = True,
) -> AsyncGenerator[str, None]:
    """Drive a local tool loop against an OpenAI-compatible remote model.

    Yields SSE ``data:`` lines (without the trailing blank line) matching the
    shapes the chat frontend already handles for local GGUF tool loops.
    """
    if permission_mode == "full":
        bypass_permissions = True
    elif bypass_permissions:
        permission_mode = "full"
    elif permission_mode not in ("ask", "auto", "off"):
        permission_mode = "ask"

    conversation = [dict(m) for m in messages]
    completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    cancel_event = cancel_event or threading.Event()

    # 9999 = "no limit" sentinel: pass None so execute_tool never times out (mirrors local GGUF loop).
    effective_tool_timeout = None if tool_call_timeout >= 9999 else tool_call_timeout
    # Cap total tool calls across all rounds at the caller's per-message budget, even with parallel calls.
    calls_remaining = max_tool_iterations
    # A forced tool_choice applies only to the first round; later rounds are freed to answer.
    active_tool_choice = tool_choice

    enabled_names = {
        (t.get("function") or {}).get("name")
        for t in tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    enabled_names.discard(None)

    # Mirror the local GGUF/safetensors loops (ToolLoopController): a repeated successful
    # call, or a render_html after it already ran, becomes an internal no-op instead of
    # re-executing. Without this a looping remote model re-applies a terminal/MCP side
    # effect every round until the budget is exhausted (#7282). tools=None keeps this to
    # duplicate/one-shot detection; the enabled-tool gate below already rejects disabled calls.
    # Thread the caller's auto_heal_tool_calls flag through (parity with the GGUF/safetensors
    # loops) so an explicit opt-out leaves malformed/scalar function.arguments unhealed instead
    # of silently recovering and executing them.
    tool_controller = ToolLoopController(tools = None, auto_heal_tool_calls = auto_heal_tool_calls)

    # Set once a terminal no-op (a repeated one-shot/duplicate) means the next pass must be a
    # tools-free synthesis, so the loop stops advertising tools and no other tool can execute.
    forced_final = False

    for _iteration in range(max_tool_iterations):
        if cancel_event.is_set():
            break

        tool_calls_acc: dict[int, dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        saw_content = False
        assistant_content = ""

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
            tools = tools,
            tool_choice = active_tool_choice,
            stream = True,
        )

        # Stop may fire while the remote read is blocked awaiting the next SSE line. Drive
        # the generator one item at a time via a cancellable task raced against the cancel
        # event so Stop abandons the read immediately (aclose() on a mid-await generator
        # raises RuntimeError, so it can't interrupt from another task).
        async def _wait_cancel() -> None:
            while not cancel_event.is_set():
                await asyncio.sleep(0.1)

        try:
            while True:
                if cancel_event.is_set():
                    break
                next_task = asyncio.ensure_future(gen.__anext__())
                cancel_task = asyncio.ensure_future(_wait_cancel())
                done, _pending = await asyncio.wait(
                    {next_task, cancel_task},
                    return_when = asyncio.FIRST_COMPLETED,
                )
                if next_task not in done:
                    # Cancelled while the read was still blocked: abandon it.
                    next_task.cancel()
                    try:
                        await next_task
                    except (asyncio.CancelledError, StopAsyncIteration, Exception):
                        pass
                    break
                cancel_task.cancel()
                try:
                    await cancel_task
                except (asyncio.CancelledError, Exception):
                    pass
                try:
                    line = next_task.result()
                except StopAsyncIteration:
                    break
                if cancel_event.is_set():
                    break
                payload = _parse_sse_data_line(line)
                if payload is None:
                    # Preserve SSE comments/fields (e.g. `: ping`, `event:`) as-is.
                    if line and line.strip() and line.strip() != "data: [DONE]":
                        yield line
                    continue
                if payload.get("error"):
                    yield line if line.startswith("data:") else f"data: {json.dumps(payload)}"
                    return

                choices = payload.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}

                content = delta.get("content")
                if content:
                    saw_content = True
                    assistant_content += str(content)
                    yield _openai_content_chunk_line(
                        completion_id = completion_id,
                        created = created,
                        model = model,
                        content = str(content),
                    )

                # Reasoning-style fields some OAI-compat servers emit.
                for key in ("reasoning_content", "reasoning"):
                    reasoning = delta.get(key)
                    if reasoning:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": str(reasoning)},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii = False)}"

                for tc_delta in delta.get("tool_calls") or []:
                    if isinstance(tc_delta, dict):
                        _merge_tool_call_delta(tool_calls_acc, tc_delta)
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass

        ordered_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc) if tool_calls_acc[i]]
        # Drop empty / nameless fragments.
        ordered_calls = [tc for tc in ordered_calls if (tc.get("function") or {}).get("name")]
        # Bound one round's fan-out before the budget slice, matching the local loops: the
        # remaining budget can be large (or the whole per-message allowance), so without this
        # a single runaway round could schedule every remaining call at once.
        ordered_calls = ordered_calls[:_MAX_TOOL_CALLS_PER_TURN]

        if not ordered_calls or finish_reason not in (None, "tool_calls", "stop"):
            # No tool calls: emit a terminal finish if we streamed content.
            if saw_content or finish_reason:
                yield _openai_content_chunk_line(
                    completion_id = completion_id,
                    created = created,
                    model = model,
                    content = "",
                    finish_reason = finish_reason or "stop",
                )
            yield "data: [DONE]"
            return

        # Enforce the per-message call budget across parallel calls in a round.
        if calls_remaining <= 0:
            if assistant_content:
                # This round's prose already streamed to the user. Keep it in the transcript
                # so the tools-free synthesis pass below continues from it instead of writing
                # the same paragraph a second time under the one already on screen.
                conversation.append({"role": "assistant", "content": assistant_content})
            break
        ordered_calls = ordered_calls[:calls_remaining]

        # Normalize ids for the conversation transcript.
        assistant_tool_calls = []
        for i, tc in enumerate(ordered_calls):
            # Include the round index so a server that omits ids does not restart at
            # call_0 every round; duplicate ids would collide in the transcript and the
            # frontend tool-event map, overwriting an earlier call's card/result.
            tc_id = tc.get("id") or f"call_{_iteration}_{i}"
            tc["id"] = tc_id
            assistant_tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments") or "{}",
                    },
                }
            )

        conversation.append(
            {
                "role": "assistant",
                # Retain streamed explanation so the follow-up round keeps the model's context.
                "content": assistant_content or None,
                "tool_calls": assistant_tool_calls,
            }
        )

        # Track how this round resolved so a round that ran nothing can stop the loop.
        handled_calls = 0
        rejected_calls = 0

        for tc in assistant_tool_calls:
            if cancel_event.is_set():
                break
            handled_calls += 1
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            arguments = _coerce_tool_arguments(raw_args)
            tool_call_id = tc["id"]

            _prov = tool_event_provenance()
            # Later rounds must be free to answer; a forced choice applies only to the first round.
            active_tool_choice = "auto"

            # Skip tools the caller did not enable (defense in depth). The check is
            # unconditional: an empty tool set means nothing is enabled, so guarding it with
            # `enabled_names and ...` would read an empty set as "no restriction" and let a
            # remote model that ignores an empty tools array run whatever it names. Nothing
            # executed, so no tool_start / tool_end card is emitted either - the local loops
            # only surface real executions (ToolCallDecision.emit_visible_events), and a card
            # for a tool the user never got makes a hallucinated name look like a real run.
            # The matching role=tool reply still goes back so the transcript stays well-formed.
            if name not in enabled_names:
                rejected_calls += 1
                result = f"Error: tool '{name}' is not enabled for this request."
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": strip_result_for_model(result),
                    }
                )
                continue

            # Duplicate / one-shot repeat gate (parity with the local loops): a call whose
            # (name, args) already succeeded, or a render_html after it already ran, is fed
            # back as an internal no-op result instead of re-executing the side effect. A
            # role=tool reply keeps every assistant tool_call matched for the next round.
            dedup_decision = tool_controller.prepare_call(tc)
            if not dedup_decision.should_execute:
                completion = tool_controller.record_noop(dedup_decision)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": strip_result_for_model(completion.result),
                    }
                )
                continue

            # Execute with the controller-normalized (healed) arguments, not the empty {}
            # that lenient JSON parsing yields on a malformed/scalar function.arguments. This
            # is the shape the dedup key and the local loops use, so a terminal/python/search
            # call the controller recovered runs with its real args instead of running empty.
            arguments = dedup_decision.arguments

            needs_confirm = bool(confirm_tool_calls) and not bypass_permissions
            if needs_confirm and permission_mode == "auto":
                # auto only pauses calls flagged unsafe; always-safe tools skip.
                if is_always_safe_tool(name) or not is_potentially_unsafe_tool_call(
                    name, arguments
                ):
                    needs_confirm = False
            elif needs_confirm and permission_mode == "off":
                needs_confirm = False

            approval_id = new_approval_id() if needs_confirm else ""
            decision_slot = begin_tool_decision(session_id, approval_id) if needs_confirm else None
            start_event: dict[str, Any] = {
                "type": "tool_start",
                "tool_name": name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "provenance": _prov,
            }
            if approval_id:
                start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirm
            result = ""
            try:
                yield f"data: {json.dumps(start_event, ensure_ascii = False)}"

                denied = False
                if needs_confirm and decision_slot is not None:
                    # Keep the stream alive while the user decides on the approval prompt.
                    decision_out: list[Any] = []
                    async for keepalive in _run_with_keepalive(
                        asyncio.to_thread(
                            wait_tool_decision,
                            decision_slot,
                            approval_id,
                            cancel_event,
                        ),
                        decision_out,
                    ):
                        yield keepalive
                    decision = decision_out[0]
                    if decision != "allow":
                        denied = True
                        result = TOOL_REJECTED_MESSAGE
                    else:
                        decision_slot = None
                if not denied:
                    try:
                        # Keep the stream alive while a slow tool runs; a search / terminal /
                        # MCP call can hold this for minutes with nothing else to send.
                        result_out: list[Any] = []
                        async for keepalive in _run_with_keepalive(
                            asyncio.to_thread(
                                execute_tool,
                                name,
                                arguments,
                                cancel_event,
                                effective_tool_timeout,
                                session_id,
                                thread_id,
                                rag_scope,
                                bypass_permissions,
                                None,
                            ),
                            result_out,
                        ):
                            yield keepalive
                        result = result_out[0]
                    except Exception as exc:
                        logger.exception("external_agentic.tool_failed name=%s", name)
                        result = f"Error executing tool '{name}': {exc}"
            finally:
                if decision_slot is not None:
                    abort_tool_decision(decision_slot, approval_id)

            model_result = result if isinstance(result, str) else str(result)
            yield f"data: {json.dumps({'type': 'tool_end', 'tool_name': name, 'tool_call_id': tool_call_id, 'result': model_result, 'provenance': _prov}, ensure_ascii = False)}"
            if not denied:
                # Only a real execution counts against the caller's tool budget; denied
                # approvals (and the controller no-ops handled above) leave the slot for a
                # later real call, matching the GGUF/safetensors _turn_executed_real_tool gate.
                calls_remaining -= 1
                # Ledger the executed result so a later identical call is deduped; a
                # failed/errored result is not marked successful, so a retry still runs.
                completion = tool_controller.record_result(dedup_decision, model_result)
                # Reuse the controller's model message so a failed tool result carries the
                # shared retry/alternate-approach nudge back to the model instead of bare error
                # text, which otherwise lets the remote model repeat the same bad call until the
                # budget is exhausted (GGUF/safetensors parity).
                conversation.append(completion.tool_message())
            else:
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": strip_result_for_model(model_result),
                    }
                )

        # Every call this round named a tool that is not enabled, so nothing ran. The model
        # has already been told so in the transcript; giving it another round just spends a
        # remote request on the same hallucination, and because a rejection costs no budget
        # it can repeat that for every remaining iteration. Go straight to the tools-free
        # synthesis instead, which is what the local loops do on a disabled call.
        if handled_calls and rejected_calls == handled_calls:
            forced_final = True
            break

        # A terminal no-op (render_html repeat / duplicate limit) declares the answer
        # final: break to the tools-free synthesis pass so a differently-named tool the
        # remote model might pick next round can no longer execute (local-loop parity).
        if tool_controller.force_final_answer:
            forced_final = True
            break

        continue

    # Budget exhausted after tool rounds: one final synthesis pass without tools.
    final_finish_reason = None
    if not cancel_event.is_set():
        # A forced-final break is a terminal no-op, not budget exhaustion, so it skips the
        # "you are out of tool budget" nudge; the model just writes its final answer.
        if max_tool_iterations > 0 and not forced_final:
            from core.inference.tool_call_parser import BUDGET_EXHAUSTED_NUDGE
            conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})
        final_gen = client.stream_chat_completion(
            messages = conversation,
            model = model,
            temperature = temperature,
            top_p = top_p,
            max_tokens = max_tokens,
            presence_penalty = presence_penalty,
            top_k = top_k,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            tools = None,
            tool_choice = "none",
            stream = True,
        )

        # Stop may fire while the remote read is blocked awaiting the next SSE line. Drive
        # the generator one item at a time via a cancellable task raced against the cancel
        # event so Stop abandons the read immediately (aclose() on a mid-await generator
        # raises RuntimeError, so it can't interrupt from another task).
        async def _wait_cancel_final() -> None:
            while not cancel_event.is_set():
                await asyncio.sleep(0.1)

        try:
            while True:
                if cancel_event.is_set():
                    break
                next_task = asyncio.ensure_future(final_gen.__anext__())
                cancel_task = asyncio.ensure_future(_wait_cancel_final())
                done, _pending = await asyncio.wait(
                    {next_task, cancel_task},
                    return_when = asyncio.FIRST_COMPLETED,
                )
                if next_task not in done:
                    # Cancelled while the read was still blocked: abandon it.
                    next_task.cancel()
                    try:
                        await next_task
                    except (asyncio.CancelledError, StopAsyncIteration, Exception):
                        pass
                    break
                cancel_task.cancel()
                try:
                    await cancel_task
                except (asyncio.CancelledError, Exception):
                    pass
                try:
                    line = next_task.result()
                except StopAsyncIteration:
                    break
                if cancel_event.is_set():
                    break
                payload = _parse_sse_data_line(line)
                if payload is None:
                    if line and line.strip() and line.strip() != "data: [DONE]":
                        yield line
                    continue
                if payload.get("error"):
                    yield line if line.startswith("data:") else f"data: {json.dumps(payload)}"
                    return

                choices = payload.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    # Surface truncation (length/content_filter) to clients.
                    final_finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}

                content = delta.get("content")
                if content:
                    yield _openai_content_chunk_line(
                        completion_id = completion_id,
                        created = created,
                        model = model,
                        content = str(content),
                    )

                for key in ("reasoning_content", "reasoning"):
                    reasoning = delta.get(key)
                    if reasoning:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": str(reasoning)},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii = False)}"
        finally:
            try:
                await final_gen.aclose()
            except Exception:
                pass

    yield _openai_content_chunk_line(
        completion_id = completion_id,
        created = created,
        model = model,
        content = "",
        finish_reason = final_finish_reason or "stop",
    )
    yield "data: [DONE]"
