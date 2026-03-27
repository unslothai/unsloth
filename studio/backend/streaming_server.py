# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Dedicated streaming server for fast SSE (Option B).

Runs as a standalone FastAPI app in a separate thread with its own event loop,
eliminating asyncio contention with the main Studio app.

Authentication: one-time tokens issued by the main app's /stream-url endpoint,
passed via the X-Stream-Token header (not in the URL to avoid logging leaks).

Supports: streaming, non-streaming, tool calling, vision, thinking mode.
Full feature parity with baseline /v1/chat/completions.

PERFORMANCE: The streaming hot path uses httpx.AsyncClient to stream directly
from llama-server, bypassing the sync generate_chat_completion() generator.
Only sends sampling parameters the client explicitly provides -- notably,
repeat_penalty defaults to llama-server's own 1.0 instead of being forced
to 1.1, which avoids a ~24% TPS penalty from repetition scanning.
"""

import asyncio
import json
import re
import threading
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from stream_token_store import consume_stream_token


stream_app = FastAPI(docs_url = None, redoc_url = None, openapi_url = None)

# ── Shared helpers ────────────────────────────────────────────


def _friendly_error(e):
    """Convert raw exception messages to user-readable strings."""
    msg = str(e)
    m = re.search(
        r"request \((\d+) tokens?\) exceeds the available context size \((\d+) tokens?\)",
        msg,
    )
    if m:
        return (
            f"Message too long: {m.group(1)} tokens exceeds the {m.group(2)}-token "
            f"context window. Try increasing the Context Length in Model settings, "
            f"or shorten the conversation."
        )
    if "Lost connection to llama-server" in msg:
        return "Lost connection to the model server. It may have crashed -- try reloading the model."
    return "An internal error occurred"


def _extract_content_parts(messages):
    """Parse messages, extracting text and image_b64 from content parts."""
    gguf_messages = []
    image_b64 = None
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url" and image_b64 is None:
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:") and "," in url:
                            image_b64 = url.split(",", 1)[1]
            content = "\n".join(text_parts) if text_parts else ""
        gguf_messages.append({"role": role, "content": content})
    return gguf_messages, image_b64


def _process_image(image_b64, llama_backend):
    """Validate and convert image to PNG if needed."""
    if image_b64 and llama_backend.is_vision:
        import base64 as _b64
        from io import BytesIO as _BytesIO
        from PIL import Image as _Image

        raw = _b64.b64decode(image_b64)
        img = _Image.open(_BytesIO(raw))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = _BytesIO()
        img.save(buf, format = "PNG")
        return _b64.b64encode(buf.getvalue()).decode("ascii")
    elif image_b64 and not llama_backend.is_vision:
        raise HTTPException(
            status_code = 400,
            detail = "Image provided but current GGUF model does not support vision.",
        )
    return image_b64


def _build_llama_payload(llama_backend, openai_messages, payload, stream = True):
    """Build the payload for llama-server /v1/chat/completions.

    Only sends repeat_penalty when the client explicitly provides it.
    This avoids the ~24% TPS penalty from repetition scanning when
    the frontend has not set a repetition penalty.
    """
    llama_payload = {
        "messages": openai_messages,
        "stream": stream,
        "temperature": payload.get("temperature", 0.6),
        "top_p": payload.get("top_p", 0.95),
        "top_k": max(payload.get("top_k", 20), 0),
        "min_p": payload.get("min_p", 0.0),
        "presence_penalty": payload.get("presence_penalty", 0.0),
    }
    # Only send repeat_penalty when the client explicitly sets repetition_penalty.
    # llama-server defaults to 1.0; forcing 1.1 costs ~24% TPS.
    if "repetition_penalty" in payload:
        llama_payload["repeat_penalty"] = payload["repetition_penalty"]
    if stream:
        llama_payload["stream_options"] = {"include_usage": True}
    if llama_backend.supports_reasoning and payload.get("enable_thinking") is not None:
        llama_payload["chat_template_kwargs"] = {
            "enable_thinking": payload["enable_thinking"]
        }
    if payload.get("max_tokens") is not None:
        llama_payload["max_tokens"] = payload["max_tokens"]
    if payload.get("stop"):
        llama_payload["stop"] = payload["stop"]
    return llama_payload


# ── CORS preflight ────────────────────────────────────────────


@stream_app.options("/stream")
async def stream_preflight():
    """Handle CORS preflight for the /stream endpoint."""
    return JSONResponse(
        content = {},
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-Stream-Token",
            "Access-Control-Max-Age": "86400",
        },
    )


# ── Request validation ────────────────────────────────────────


async def _validate_request(request: Request):
    """Validate token, parse body, get backend. Returns all needed context."""
    token = request.headers.get("X-Stream-Token")
    if not token:
        raise HTTPException(status_code = 401, detail = "Missing X-Stream-Token header")
    username = consume_stream_token(token)
    if username is None:
        raise HTTPException(status_code = 401, detail = "Invalid or expired stream token")

    body_bytes = await request.body()
    try:
        payload = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code = 400, detail = "Invalid JSON body")

    from routes.inference import get_llama_cpp_backend

    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(status_code = 400, detail = "No GGUF model loaded")

    messages = payload.get("messages", [])
    gguf_messages, image_b64 = _extract_content_parts(messages)

    # Legacy image_base64 fallback
    if not image_b64:
        image_b64 = payload.get("image_base64")

    # Image validation and conversion
    image_b64 = _process_image(image_b64, llama_backend)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = llama_backend.model_identifier or "unknown"

    return (
        payload,
        llama_backend,
        gguf_messages,
        image_b64,
        completion_id,
        created,
        model_name,
    )


# ── Path A: Direct async streaming (HOT PATH) ────────────────


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
}

_SEP = (",", ":")


async def _handle_async_stream(
    request,
    payload,
    llama_backend,
    gguf_messages,
    image_b64,
    completion_id,
    created,
    model_name,
):
    """
    Stream directly from llama-server using httpx.AsyncClient.

    Bypasses the sync generate_chat_completion() generator and its
    asyncio.to_thread overhead.  llama-server speaks standard OpenAI SSE
    with delta tokens natively, so no cumulative-to-delta conversion needed.
    """
    openai_messages = llama_backend._build_openai_messages(gguf_messages, image_b64)
    llama_payload = _build_llama_payload(
        llama_backend, openai_messages, payload, stream = True
    )

    port = llama_backend._port
    api_key = llama_backend._api_key
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    timeout = httpx.Timeout(connect = 30, read = 120.0, write = 10, pool = 10)

    async def sse_generator():
        try:
            # Role chunk
            role = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(role, separators = _SEP)}\n\n"

            async with httpx.AsyncClient(timeout = timeout) as client:
                async with client.stream(
                    "POST", url, json = llama_payload, headers = headers
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        raise RuntimeError(
                            f"llama-server returned {resp.status_code}: {error_body.decode()}"
                        )

                    buffer = ""
                    in_thinking = False
                    has_content_tokens = False
                    reasoning_text = ""
                    stream_usage = None
                    stream_timings = None
                    stream_done = False

                    async for raw_chunk in resp.aiter_text():
                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                if in_thinking:
                                    if has_content_tokens:
                                        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]}, separators = _SEP)}\n\n"
                                    else:
                                        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': reasoning_text}, 'finish_reason': None}]}, separators = _SEP)}\n\n"
                                stream_done = True
                                break
                            if not line.startswith("data: "):
                                continue

                            try:
                                data = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue

                            _t = data.get("timings")
                            if _t:
                                stream_timings = _t
                            _u = data.get("usage")
                            if _u:
                                stream_usage = _u

                            choices = data.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            # Handle reasoning_content -> <think> tags
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                reasoning_text += reasoning
                                if not in_thinking:
                                    in_thinking = True
                                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': '<think>'}, 'finish_reason': None}]}, separators = _SEP)}\n\n"
                                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': reasoning}, 'finish_reason': None}]}, separators = _SEP)}\n\n"

                            # Handle content tokens
                            token = delta.get("content", "")
                            if token:
                                has_content_tokens = True
                                if in_thinking:
                                    in_thinking = False
                                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': '</think>'}, 'finish_reason': None}]}, separators = _SEP)}\n\n"
                                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]}, separators = _SEP)}\n\n"

                        if stream_done:
                            break

            # Final stop chunk
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final, separators = _SEP)}\n\n"

            # Usage chunk
            if stream_usage or stream_timings:
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": (stream_usage or {}).get("prompt_tokens", 0),
                        "completion_tokens": (stream_usage or {}).get(
                            "completion_tokens", 0
                        ),
                        "total_tokens": (stream_usage or {}).get("total_tokens", 0),
                    },
                }
                if stream_timings:
                    usage_chunk["timings"] = stream_timings
                yield f"data: {json.dumps(usage_chunk, separators = _SEP)}\n\n"

            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield f"data: {json.dumps({'error': {'message': _friendly_error(e), 'type': 'server_error'}})}\n\n"

    return StreamingResponse(
        sse_generator(), media_type = "text/event-stream", headers = _SSE_HEADERS
    )


# ── Path B: Tool calling (asyncio.to_thread) ─────────────────


async def _handle_tool_stream(
    request,
    payload,
    llama_backend,
    gguf_messages,
    image_b64,
    completion_id,
    created,
    model_name,
):
    """Handle a tool-calling streaming request via asyncio.to_thread.

    Tool execution is the bottleneck, not streaming, so the thread overhead
    is acceptable here.
    """
    from core.inference.tools import ALL_TOOLS

    cancel_event = threading.Event()

    p_enabled_tools = payload.get("enabled_tools")
    if p_enabled_tools is not None:
        tools_to_use = [
            t for t in ALL_TOOLS if t["function"]["name"] in p_enabled_tools
        ]
    else:
        tools_to_use = ALL_TOOLS

    _sentinel = object()

    async def tool_sse():
        try:
            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(first, separators = _SEP)}\n\n"

            gen = llama_backend.generate_chat_completion_with_tools(
                messages = gguf_messages,
                tools = tools_to_use,
                temperature = payload.get("temperature", 0.6),
                top_p = payload.get("top_p", 0.95),
                top_k = payload.get("top_k", 20),
                min_p = payload.get("min_p", 0.01),
                max_tokens = payload.get("max_tokens"),
                repetition_penalty = payload.get("repetition_penalty", 1.1),
                presence_penalty = payload.get("presence_penalty", 0.0),
                cancel_event = cancel_event,
                enable_thinking = payload.get("enable_thinking"),
                auto_heal_tool_calls = payload.get("auto_heal_tool_calls", True),
                max_tool_iterations = payload.get("max_tool_calls_per_message", 10),
                tool_call_timeout = payload.get("tool_call_timeout", 300),
                session_id = payload.get("session_id"),
            )

            prev_text = ""
            _usage = None
            _timings = None

            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                event = await asyncio.to_thread(next, gen, _sentinel)
                if event is _sentinel:
                    break
                if event["type"] == "status":
                    yield f"data: {json.dumps({'type': 'tool_status', 'content': event['text']})}\n\n"
                    continue
                if event["type"] in ("tool_start", "tool_end"):
                    yield f"data: {json.dumps(event)}\n\n"
                    continue
                if event["type"] == "metadata":
                    _usage = event.get("usage")
                    _timings = event.get("timings")
                    continue
                cumulative = event.get("text", "")
                new_text = cumulative[len(prev_text) :]
                prev_text = cumulative
                if not new_text:
                    continue
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": new_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, separators = _SEP)}\n\n"

            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final, separators = _SEP)}\n\n"

            if _usage or _timings:
                uc = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": (_usage or {}).get("prompt_tokens", 0),
                        "completion_tokens": (_usage or {}).get("completion_tokens", 0),
                        "total_tokens": (_usage or {}).get("total_tokens", 0),
                    },
                }
                if _timings:
                    uc["timings"] = _timings
                yield f"data: {json.dumps(uc, separators = _SEP)}\n\n"

            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        except Exception as e:
            yield f"data: {json.dumps({'error': {'message': _friendly_error(e), 'type': 'server_error'}})}\n\n"

    return StreamingResponse(
        tool_sse(), media_type = "text/event-stream", headers = _SSE_HEADERS
    )


# ── Path C: Non-streaming ─────────────────────────────────────


async def _handle_non_streaming(
    payload, llama_backend, gguf_messages, image_b64, completion_id, created, model_name
):
    """Handle a non-streaming request. Returns a JSON response."""
    cancel_event = threading.Event()

    def _run_sync():
        gen = llama_backend.generate_chat_completion(
            messages = gguf_messages,
            image_b64 = image_b64,
            temperature = payload.get("temperature", 0.6),
            top_p = payload.get("top_p", 0.95),
            top_k = payload.get("top_k", 20),
            min_p = payload.get("min_p", 0.01),
            max_tokens = payload.get("max_tokens"),
            repetition_penalty = payload.get("repetition_penalty", 1.0),
            presence_penalty = payload.get("presence_penalty", 0.0),
            stop = payload.get("stop"),
            cancel_event = cancel_event,
            enable_thinking = payload.get("enable_thinking"),
        )
        text = ""
        usage = None
        timings = None
        for item in gen:
            if isinstance(item, dict) and item.get("type") == "metadata":
                usage = item.get("usage")
                timings = item.get("timings")
            elif isinstance(item, str):
                text = item
        return text, usage, timings

    text, usage, timings = await asyncio.to_thread(_run_sync)

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
            "completion_tokens": (usage or {}).get("completion_tokens", 0),
            "total_tokens": (usage or {}).get("total_tokens", 0),
        },
    }
    if timings:
        result["timings"] = timings

    return JSONResponse(
        content = result,
        headers = {"Access-Control-Allow-Origin": "*"},
    )


# ── Main endpoint ─────────────────────────────────────────────


@stream_app.post("/stream")
async def stream_endpoint(request: Request):
    """
    Stream chat completions with minimal overhead.

    Three paths:
    - Path A (hot): async httpx streaming direct to llama-server
    - Path B: tool calling via asyncio.to_thread (tool exec is the bottleneck)
    - Path C: non-streaming one-shot JSON response
    """
    (
        payload,
        llama_backend,
        gguf_messages,
        image_b64,
        completion_id,
        created,
        model_name,
    ) = await _validate_request(request)

    # Path C: Non-streaming
    stream = payload.get("stream", True)
    if not stream:
        return await _handle_non_streaming(
            payload,
            llama_backend,
            gguf_messages,
            image_b64,
            completion_id,
            created,
            model_name,
        )

    # Path B: Tool calling
    use_tools = payload.get("use_tools", False)
    if use_tools and llama_backend.supports_tools:
        return await _handle_tool_stream(
            request,
            payload,
            llama_backend,
            gguf_messages,
            image_b64,
            completion_id,
            created,
            model_name,
        )

    # Path A: Direct async streaming (hot path)
    return await _handle_async_stream(
        request,
        payload,
        llama_backend,
        gguf_messages,
        image_b64,
        completion_id,
        created,
        model_name,
    )


# ── Server lifecycle ──────────────────────────────────────────


def start_streaming_server(port: int) -> None:
    """Start the streaming server in the current thread (blocking). Use in a daemon thread."""
    import uvicorn

    uvicorn.run(
        stream_app,
        host = "127.0.0.1",
        port = port,
        log_level = "warning",
        access_log = False,
    )


def find_free_port() -> int:
    """Find a free TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
