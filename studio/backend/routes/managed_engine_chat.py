# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native managed-engine tool calls and Studio's shared execution loop."""

import asyncio
import json
import threading
import time
import uuid
from contextlib import aclosing

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from core.inference.engine_transport import engine_messages, engine_request_timeout
from core.inference.inference_gate import gated_async_client
from core.inference.managed_engine import fold_reasoning_into_answer
from core.inference.http_stream import closing_response_lines
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools
from utils.api_errors import openai_error_body


class ManagedToolTransport:
    heals_text_tool_calls = True
    sanitizes_provider_frames = False

    def __init__(
        self,
        client,
        engine,
        body,
        *,
        preserve_thinking = None,
        message_fitter = None,
        continuing = False,
    ):
        self.client, self.engine, self.body = client, engine, body
        self._continuing = continuing
        self._preserve_thinking = preserve_thinking
        self._message_fitter = message_fitter
        # The tool loop keeps a turn's reasoning_content on the assistant message only when asked.
        self.preserves_reasoning = bool(engine.reasoning_parser and preserve_thinking)

    async def stream(self, *, messages, tools, tool_choice, cancel_event):
        from routes import inference as api

        truncation_line = max_tokens = None
        if self._message_fitter is not None:
            messages, max_tokens, truncation_line = await self._message_fitter(messages)
        body = {
            **self.body,
            "messages": self.engine.replay_reasoning(
                engine_messages(messages), self._preserve_thinking
            ),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        continuing = self._continuing and messages[-1].get("role") == "assistant"
        if continuing:
            self.engine.continue_final_message(body)
        if truncation_line:
            yield truncation_line
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        else:
            body.pop("tools", None)
            body.pop("tool_choice", None)
        if cancel_event.is_set():
            return
        async with self.client.stream(
            "POST",
            self.engine.base_url + "/v1/chat/completions",
            headers = self.engine.headers,
            json = body,
        ) as response:
            if response.status_code != 200:
                await response.aread()
                raise api._openai_passthrough_error(response.status_code, response.text)
            lines = closing_response_lines(response)
            try:
                async for line in lines:
                    if cancel_event.is_set():
                        return
                    if line.startswith("data:"):
                        yield _as_answer(line) if continuing else _with_reasoning_content(line)
            finally:
                await lines.aclose()


def _as_answer(line: str) -> str:
    try:
        event = json.loads(line[5:])
    except ValueError:
        return line
    for choice in event.get("choices") or () if isinstance(event, dict) else ():
        if isinstance(choice, dict) and isinstance(choice.get("delta"), dict):
            fold_reasoning_into_answer(choice["delta"])
    return "data: " + json.dumps(event)


def _with_reasoning_content(line: str) -> str:
    """vLLM 0.11+ streams thinking as ``reasoning``; Studio's readers take ``reasoning_content``."""
    if '"reasoning"' not in line:
        return line
    try:
        event = json.loads(line[5:])
    except ValueError:
        return line
    changed = False
    for choice in event.get("choices") or () if isinstance(event, dict) else ():
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if isinstance(delta, dict) and "reasoning" in delta:
            reasoning = delta.pop("reasoning")
            if reasoning and not delta.get("reasoning_content"):
                delta["reasoning_content"] = reasoning
            changed = True
    return "data: " + json.dumps(event) if changed else line


def _message_fitter(
    payload,
    engine,
    public,
    tools,
    studio_loop = False,
):
    """Auto-compact to the managed engine's window; the exact count refuses tools, so this estimates."""
    from routes import inference as api

    if (
        getattr(engine, "engine", None) not in ("vllm", "sglang")
        or api._rolling_context_policy(payload) is None
        or not engine.context
    ):
        return None
    fit_payload = payload.model_copy(
        update = {
            "context_window": engine.context,
            "compaction_threshold": min(
                api._EXTERNAL_COMPACTION_THRESHOLD_MAX,
                int(engine.context * (1 - api._EXTERNAL_COMPACTION_HEADROOM)),
            ),
        }
    )

    # The first fit sees the client's branch; later ones see the loop's conversation, which can end
    # on a controller re-prompt that recall must not query for.
    branch: list = []
    tool_loop = studio_loop and not api._tool_calls_are_disabled(payload)
    # An uncapped reply is sized by the engine; a cap from the estimate can push the prompt over.
    limit = api._effective_max_tokens(payload)
    capped = bool(limit) and limit < engine.context

    async def fit(messages):
        if not branch:
            branch.extend(messages)
        fitted, truncation, max_tokens = await asyncio.to_thread(
            api._fit_external_context,
            messages,
            fit_payload,
            tools = tools,
            branch = branch,
            tool_loop = tool_loop,
        )
        line = None
        if truncation and truncation.get("dropped_messages"):
            line = api._context_truncated_sse_chunk(
                "chatcmpl-" + uuid.uuid4().hex[:12], public, truncation
            ).rstrip("\n")
        return fitted, max_tokens if capped else None, line

    return fit


def _with_model(line: str, public: str) -> str:
    if not line.startswith("data:") or '"model"' not in line:
        return line
    try:
        event = json.loads(line[5:])
    except ValueError:
        return line
    if not isinstance(event, dict) or event.get("model") in (None, public):
        return line
    event["model"] = public
    return "data: " + json.dumps(event)


async def managed_tool_chat(
    payload,
    request,
    backend,
    messages,
    system_prompt,
    monitor_id,
    *,
    promoted_image_parts = (),
):
    from routes import inference as api
    from state.tool_policy import get_tool_policy
    from core.inference.sse_control_frames import sanitize_provider_sse_line

    tools_on = bool(api._effective_enable_tools(payload))
    ui_events = api._ui_stream_events_enabled(request)
    if tools_on and not api._launcher_tool_default_applies(payload, ui_events):
        tools_on = False
    if api._response_format_constrains_decoding(
        payload
    ) and not api._explicit_studio_tool_loop_requested(payload):
        tools_on = False
    if payload.tool_choice == "none":
        tools_on = False
    mcp_on = (
        payload.tool_choice != "none"
        and bool(payload.mcp_enabled)
        and get_tool_policy() is not False
    )
    has_history = any(m.role == "tool" or m.tool_calls for m in payload.messages)
    info = backend.models.get(backend.active_model_name, {})
    # A catalog sent with tool_choice "none" and no tool turns cannot produce a call, so a model
    # without a parser serves it as plain chat; one with a parser still renders the catalog.
    idle_catalog = payload.tool_choice == "none" and not has_history
    client_tools = None if idle_catalog and not info.get("supports_tools") else payload.tools
    if not (tools_on or mcp_on or client_tools or has_history):
        return None

    def reject(message):
        api.api_monitor.fail(monitor_id, message)
        return HTTPException(status_code = 400, detail = message)

    if not info.get("supports_tools"):
        raise reject(
            "No native tool parser is configured for this model's chat template. Use Default for tool calling with this model."
        )
    tools = (
        await api._select_request_tools(
            payload,
            tools_on = tools_on,
            mcp_allowed = mcp_on,
            supports_vision = bool(info.get("is_vision")),
            ui_events = ui_events,
        )
        if tools_on or mcp_on
        else []
    )
    if tools and api._wants_multiple_choices(payload):
        raise reject("Multiple choices are unavailable when Unsloth executes tools.")
    if tools and api._response_format_constrains_decoding(payload):
        # The tool loop never forwards response_format; refuse rather than violate it (as GGUF does).
        message = (
            "response_format is not supported with Unsloth tool execution; send the request "
            "without enable_tools or mcp_enabled to use guided decoding."
        )
        api.api_monitor.fail(monitor_id, message)
        raise HTTPException(
            status_code = 400,
            detail = openai_error_body(
                message, status = 400, code = "unsupported_parameter", param = "response_format"
            ),
        )
    if tools:
        api._reject_confirm_gate_without_channel(
            payload, ui_events, monitor_id, api._catalog_names(tools)
        )
    selected = tools or payload.tools or []
    from core.inference.chat_template_helpers import forced_tool_name

    forced = forced_tool_name(payload.tool_choice)
    if forced and forced not in api._catalog_names(selected):
        raise reject(f"Tool '{forced}' is not enabled.")
    messages = api._set_or_prepend_system_message(messages, system_prompt)
    if tools:
        nudge = api._build_tool_action_nudge(
            tools = tools,
            model_name = backend.active_model_name,
            full_access = bool(payload.bypass_permissions),
        )
        nudge = await api._apply_rag_nudge(nudge, tools, rag_scope = payload.rag_scope)
        if not api._tool_calls_are_disabled(payload):
            nudge = api._apply_compaction_nudge(nudge, tools, payload = payload)
        if nudge:
            messages = api._append_to_system_message(messages, nudge)
    try:
        messages = engine_messages(messages)
    except ValueError as exc:
        raise reject(str(exc)) from exc
    engine = backend._managed_engine
    # Serves config.identifier (a local path for directory loads); clients get the same public id.
    public = api._orchestrator_public_model_id(backend) or engine.model
    messages = engine.replay_reasoning(messages, payload.preserve_thinking)
    body = {
        "model": engine.model,
        "messages": messages,
        **engine.reasoning_fields(
            payload.enable_thinking, payload.reasoning_effort, payload.preserve_thinking
        ),
    }
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "logit_bias",
        "parallel_tool_calls",
        "n",
    ):
        value = getattr(payload, key, None)
        if value is not None:
            body[key] = value
    stop = api._normalize_stop_sequences(payload.stop)
    if stop:
        body["stop"] = stop
    limit = api._effective_max_tokens(payload)
    if limit and 0 < limit < engine.context:
        body["max_tokens"] = limit
    if payload.tools and not tools:
        body["tools"] = payload.tools
    if payload.tool_choice is not None and not tools:
        body["tool_choice"] = payload.tool_choice
    response_format = api._response_format_beside_tools(payload, tools or payload.tools)
    if response_format is not None:
        body["response_format"] = response_format
    continuing = api._continue_final_message(payload)

    cancel = api._chat_cancel_event(request)
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    tracker = api._TrackedCancel.for_payload(
        cancel, payload, payload.cancel_id, payload.session_id, completion_id
    )
    tracker.__enter__()

    async def events():
        client = gated_async_client(trust_env = False, timeout = engine_request_timeout())
        watcher = asyncio.create_task(
            api._await_cancel_or_disconnect_then_close_client(
                cancel_event = cancel, request = request, client = client
            )
        )
        transport = ManagedToolTransport(
            client,
            engine,
            body,
            preserve_thinking = payload.preserve_thinking,
            message_fitter = _message_fitter(
                payload, engine, public, tools or payload.tools, studio_loop = bool(tools)
            ),
            continuing = continuing,
        )
        stripper = api.ServerToolCallStripper()
        stream = None
        failed = False
        try:
            if tools:
                stream = stream_with_studio_tools(
                    transport,
                    run = ToolLoopRun(
                        messages = messages,
                        session_id = payload.session_id,
                        thread_id = payload.thread_id,
                        model = engine.model,
                        tool_choice = payload.tool_choice,
                        supports_vision = bool(info.get("is_vision")),
                        promoted_image_parts = promoted_image_parts,
                        continue_final_message = continuing,
                    ),
                    policy = ToolLoopPolicy(
                        tools = tools,
                        max_calls = payload.max_tool_calls_per_message
                        if payload.max_tool_calls_per_message is not None
                        else 25,
                        timeout = payload.tool_call_timeout or 300,
                        permission_mode = payload.permission_mode or "auto",
                        sandbox_level = payload.sandbox_level,
                        confirm_calls = api._permission_mode_confirm(payload)
                        or api._off_mode_sandbox_gate(payload, ui_events),
                        bypass_permissions = bool(payload.bypass_permissions),
                        rag_scope = payload.rag_scope,
                        website_policy = getattr(payload, "_website_policy", None),
                        auto_heal = payload.auto_heal_tool_calls,
                        nudge_tool_calls = payload.nudge_tool_calls,
                        deduplicate_tool_calls = payload.deduplicate_tool_calls,
                        on_withheld_tool_call = None if ui_events else stripper.arm,
                        on_provider_turn_end = None if ui_events else stripper.end_turn,
                    ),
                    cancel_event = cancel,
                )
            elif not payload.stream:
                request_body = {**body, "stream": False}
                truncation_line = None
                if transport._message_fitter is not None:
                    fitted, max_tokens, truncation_line = await transport._message_fitter(messages)
                    request_body["messages"] = engine.replay_reasoning(
                        engine_messages(fitted), payload.preserve_thinking
                    )
                    if max_tokens:
                        request_body["max_tokens"] = max_tokens
                if continuing:
                    engine.continue_final_message(request_body)
                response = await client.post(
                    engine.base_url + "/v1/chat/completions",
                    headers = engine.headers,
                    json = request_body,
                )
                if response.status_code != 200:
                    raise api._openai_passthrough_error(response.status_code, response.text)
                data = response.json()
                if isinstance(data, dict) and "model" in data:
                    data["model"] = public
                for choice in data.get("choices") or () if isinstance(data, dict) else ():
                    message = choice.get("message") if isinstance(choice, dict) else None
                    if continuing and isinstance(message, dict):
                        fold_reasoning_into_answer(message)
                    if isinstance(message, dict) and "reasoning" in message:
                        reasoning = message.pop("reasoning")
                        if reasoning and not message.get("reasoning_content"):
                            message["reasoning_content"] = reasoning
                if truncation_line and isinstance(data, dict):
                    data["context_truncated"] = json.loads(truncation_line[5:])["context_truncated"]
                api._monitor_openai_chunk(monitor_id, data, streaming = False)
                yield data
                return
            else:
                stream = transport.stream(
                    messages = messages,
                    tools = payload.tools,
                    tool_choice = payload.tool_choice,
                    cancel_event = cancel,
                )
            async for line in stream:
                if not tools:
                    line = sanitize_provider_sse_line(line)
                    if line is None:
                        continue
                if api._monitor_openai_sse_line(monitor_id, line) == "error":
                    failed = True
                if not ui_events and api.is_ui_control_sse_line(line):
                    yield ": keep-alive"
                    continue
                if tools and not ui_events:
                    line = stripper.strip(line)
                    if line is None:
                        continue
                if (
                    payload.stream
                    and not api._wants_stream_usage(payload)
                    and api._is_openai_usage_only_sse(line)
                ):
                    continue
                yield _with_model(line, public)
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception as exc:
            if cancel.is_set():
                raise asyncio.CancelledError() from exc
            failed = True
            api.api_monitor.fail(monitor_id, "Engine tool request failed.")
            raise
        finally:
            try:
                if stream is not None:
                    await stream.aclose()
            finally:
                try:
                    await api._stop_local_disconnect_cancel_watcher(watcher)
                    await client.aclose()
                finally:
                    try:
                        if not failed:
                            api.api_monitor.finish(
                                monitor_id, "cancelled" if cancel.is_set() else "completed"
                            )
                    finally:
                        tracker.__exit__(None, None, None)

    if payload.stream:

        async def unstarted_cleanup():
            cancel.set()
            try:
                api.api_monitor.finish(monitor_id, "cancelled")
            finally:
                tracker.__exit__(None, None, None)

        async def sse():
            stream = events()
            try:
                async for line in stream:
                    yield line + "\n\n"
            except HTTPException as exc:
                yield api._openai_stream_error_sse(
                    exc.detail
                    if isinstance(exc.detail, dict)
                    else {"error": {"message": str(exc.detail)}}
                )
            except Exception:
                yield api._openai_stream_error_sse(
                    {"error": {"message": "Engine tool request failed."}}
                )
            finally:
                await stream.aclose()

        return api._SameTaskStreamingResponse(
            sse(),
            media_type = "text/event-stream",
            headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            unstarted_cleanup = unstarted_cleanup,
        )

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": public,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
        ],
    }
    async with aclosing(events()) as stream:
        async for line in stream:
            if isinstance(line, dict):
                return JSONResponse(line)
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("error"):
                raise HTTPException(status_code = 502, detail = event)
            if event.get("usage"):
                result["usage"] = event["usage"]
            if event.get("context_truncated"):
                result["context_truncated"] = event["context_truncated"]
            message = result["choices"][0]["message"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                message["content"] += delta.get("content") or ""
                if delta.get("reasoning_content"):
                    message["reasoning_content"] = (
                        message.get("reasoning_content", "") + delta["reasoning_content"]
                    )
                if choice.get("finish_reason"):
                    result["choices"][0]["finish_reason"] = choice["finish_reason"]
    return JSONResponse(result)


async def managed_engine_completions(request, current_subject, backend, body):
    from routes import inference as api
    from core.inference.llama_keepwarm import mark_response_failed

    engine = backend._managed_engine
    public = api._orchestrator_public_model_id(backend) or engine.model
    stream = bool(body.get("stream"))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    upstream = {**body, "model": engine.model}
    if stream:
        # Always asked for so the monitor counts tokens; dropped again unless the caller opted in.
        upstream["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
    monitor_id = api.api_monitor.start(
        endpoint = request.url.path,
        via_api_key = api._request_used_api_key(request),
        method = request.method,
        model = public,
        prompt = api._flatten_monitor_prompt(body.get("prompt", "")),
        context_length = engine.context,
        subject = current_subject,
    )
    cancel = threading.Event()
    tracker = api._TrackedCancel(cancel, model = public, kind = "completions")
    tracker.__enter__()
    client = gated_async_client(trust_env = False, timeout = engine_request_timeout())
    watcher = asyncio.create_task(
        api._await_cancel_or_disconnect_then_close_client(
            cancel_event = cancel, request = request, client = client
        )
    )
    scope = getattr(request, "scope", None)

    async def close():
        try:
            await api._stop_local_disconnect_cancel_watcher(watcher)
            await client.aclose()
        finally:
            tracker.__exit__(None, None, None)

    url = engine.base_url + "/v1/completions"
    try:
        if stream:
            response = await client.send(
                client.build_request("POST", url, headers = engine.headers, json = upstream),
                stream = True,
            )
            if response.status_code != 200:
                await response.aread()
        else:
            response = await client.post(url, headers = engine.headers, json = upstream)
    except asyncio.CancelledError:
        cancel.set()
        await close()
        api.api_monitor.finish(monitor_id, "cancelled")
        raise
    except Exception as exc:
        await close()
        if cancel.is_set():
            mark_response_failed(scope)
            api.api_monitor.finish(monitor_id, "cancelled")
            raise HTTPException(status_code = 499, detail = "Request cancelled.") from exc
        api.api_monitor.fail(monitor_id, api._friendly_error(exc))
        raise
    if response.status_code != 200:
        await close()
        api.api_monitor.fail(monitor_id, response.text[:500])
        raise api._openai_passthrough_error(response.status_code, response.text)

    if not stream:
        await close()
        data = response.json()
        if isinstance(data, dict) and "model" in data:
            data["model"] = public
        try:
            api._monitor_openai_chunk(monitor_id, data, engine.context)
        except Exception:
            pass
        api.api_monitor.finish(monitor_id)
        return JSONResponse(data)

    async def relay():
        lines = closing_response_lines(response)
        try:
            async for line in lines:
                if not line.startswith("data:"):
                    continue
                if api._monitor_openai_sse_line(monitor_id, line, engine.context) == "error":
                    mark_response_failed(scope)
                out = api._cmpl_stream_event_out(_with_model(line, public).encode(), include_usage)
                if out is not None:
                    yield out + b"\n\n"
            if cancel.is_set():
                mark_response_failed(scope)
                api.api_monitor.finish(monitor_id, "cancelled")
            else:
                api.api_monitor.finish(monitor_id)
        except asyncio.CancelledError:
            cancel.set()
            api.api_monitor.finish(monitor_id, "cancelled")
            raise
        except Exception as exc:
            mark_response_failed(scope)
            if cancel.is_set():
                api.api_monitor.finish(monitor_id, "cancelled")
                return
            api.api_monitor.fail(monitor_id, api._friendly_error(exc))
            yield api._openai_stream_error_sse_bytes(api._openai_stream_error_chunk(exc))
        finally:
            try:
                await lines.aclose()
            finally:
                await close()

    async def unstarted_cleanup():
        cancel.set()
        try:
            # close() releases the active-generation entry even when aclose fails.
            try:
                await response.aclose()
            finally:
                await close()
        finally:
            api.api_monitor.finish(monitor_id, "cancelled")

    return api._sse_streaming_response(
        relay(), unstarted_cleanup = unstarted_cleanup, monitor_id = monitor_id
    )
