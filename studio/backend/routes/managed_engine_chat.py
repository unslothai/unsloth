# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native managed-engine tool calls and Studio's shared execution loop."""

import asyncio
import json
import time
import uuid
from contextlib import aclosing

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from core.inference.engine_transport import engine_request_timeout
from core.inference.http_stream import closing_response_lines
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools


class ManagedToolTransport:
    heals_text_tool_calls = True
    sanitizes_provider_frames = False

    def __init__(self, client, engine, body):
        self.client, self.engine, self.body = client, engine, body

    async def stream(self, *, messages, tools, tool_choice, cancel_event):
        from routes import inference as api

        body = {
            **self.body,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
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
                        yield line
            finally:
                await lines.aclose()


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
    if payload.tool_choice == "none":
        tools_on = False
    mcp_on = (
        payload.tool_choice != "none"
        and bool(payload.mcp_enabled)
        and get_tool_policy() is not False
    )
    has_history = any(m.role == "tool" or m.tool_calls for m in payload.messages)
    if not (tools_on or mcp_on or payload.tools or has_history):
        return None

    def reject(message):
        api.api_monitor.fail(monitor_id, message)
        return HTTPException(status_code = 400, detail = message)

    info = backend.models.get(backend.active_model_name, {})
    if not info.get("supports_tools"):
        raise reject(
            "No native tool parser is configured for this model's chat template. Use Default for tool calling with this model."
        )
    tools = (
        await api._select_request_tools(payload, tools_on = tools_on, mcp_allowed = mcp_on)
        if tools_on or mcp_on
        else []
    )
    if tools and api._wants_multiple_choices(payload):
        raise reject("Multiple choices are unavailable when Studio executes tools.")
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
        if nudge:
            messages = api._append_to_system_message(messages, nudge)
    engine = backend._managed_engine
    body = {
        "model": engine.model,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
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

    cancel = api._chat_cancel_event(request)
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    tracker = api._TrackedCancel.for_payload(
        cancel, payload, payload.cancel_id, payload.session_id, completion_id
    )
    tracker.__enter__()

    async def events():
        client = httpx.AsyncClient(trust_env = False, timeout = engine_request_timeout())
        watcher = asyncio.create_task(
            api._await_cancel_or_disconnect_then_close_client(
                cancel_event = cancel, request = request, client = client
            )
        )
        transport = ManagedToolTransport(client, engine, body)
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
                    ),
                    policy = ToolLoopPolicy(
                        tools = tools,
                        max_calls = payload.max_tool_calls_per_message
                        if payload.max_tool_calls_per_message is not None
                        else 25,
                        timeout = payload.tool_call_timeout or 300,
                        permission_mode = payload.permission_mode or "auto",
                        confirm_calls = api._permission_mode_confirm(payload),
                        bypass_permissions = bool(payload.bypass_permissions),
                        rag_scope = payload.rag_scope,
                        auto_heal = payload.auto_heal_tool_calls,
                        nudge_tool_calls = payload.nudge_tool_calls,
                        on_withheld_tool_call = None if ui_events else stripper.arm,
                        on_provider_turn_end = None if ui_events else stripper.end_turn,
                    ),
                    cancel_event = cancel,
                )
            elif not payload.stream:
                response = await client.post(
                    engine.base_url + "/v1/chat/completions",
                    headers = engine.headers,
                    json = {**body, "stream": False},
                )
                if response.status_code != 200:
                    raise api._openai_passthrough_error(response.status_code, response.text)
                data = response.json()
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
                yield line
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
        "model": engine.model,
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
            for choice in event.get("choices", []):
                result["choices"][0]["message"]["content"] += (
                    choice.get("delta", {}).get("content") or ""
                )
                if choice.get("finish_reason"):
                    result["choices"][0]["finish_reason"] = choice["finish_reason"]
    return JSONResponse(result)
