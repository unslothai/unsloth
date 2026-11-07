# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One live Claude Code session per tool loop; Studio's tools are an in-process MCP server whose handlers wait for the loop's result."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator
from functools import partial
from typing import Any, Callable, Optional

from core.inference.claude_agent import (
    CLI_MISSING,
    EFFORTS,
    PROVIDER_TYPE,
    SDK_MISSING,
    TOOL_PREFIX,
    TOOL_SERVER,
    _options,
    _sdk,
    _text,
    _user_blocks,
    build_prompt,
    clamp_effort,
    cli_version,
    ensure_catalog,
    failure,
    find_cli,
    image_url,
    outdated_message,
    remember_models,
    thinking_config,
)
from core.inference.external_provider import _build_usage_chunk, _error_sse_line

_CANCEL_POLL_S = 0.05
_NAME_LIMIT = 64 - len(TOOL_PREFIX)
_TOOLS_CLOSED = (
    "Unsloth did not run this call: tools are not available for the rest of this reply. "
    "Answer without calling a tool."
)
_NOT_RUN = "Unsloth did not run this tool call."
_CANCELLED = "Cancelled by the user."
_ERROR_PREFIXES = (
    "Unsloth did not",
    "Unsloth stopped",
    "The user declined",
    "This tool call was not run",
    "Error: tool raised",
)
_DONE = "data: [DONE]"


class _Ended:
    def __init__(self, error: Optional[BaseException]) -> None:
        self.error = error


def tool_name(name: str) -> str:
    if len(name) <= _NAME_LIMIT and re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:10]
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[: _NAME_LIMIT - 11] + "_" + digest


def _is_error(text: str) -> bool:
    from core.inference.tool_call_parser import TOOL_ERROR_NUDGE
    return text.endswith(TOOL_ERROR_NUDGE) or text.startswith(_ERROR_PREFIXES)


def _mcp_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for part in content if isinstance(content, list) else []:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
            continue
        url = image_url(part)
        if url is None:
            continue
        header, _, data = url.partition(",")
        if url.startswith("data:image/") and header.endswith(";base64") and data:
            blocks.append({"type": "image", "data": data, "mimeType": header[5:].split(";")[0]})
        else:
            blocks.append(
                {"type": "text", "text": "[An image was left out: it is not inline data.]"}
            )
    return blocks


def _appended(old: Any, new: Any) -> Any:
    if isinstance(old, str) and isinstance(new, str) and new.startswith(old):
        return new[len(old) :]
    if isinstance(old, list) and isinstance(new, list) and new[: len(old)] == old:
        return new[len(old) :]
    return None


async def _await_cancel(cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(_CANCEL_POLL_S)


class ClaudeAgentTransport:
    heals_text_tool_calls = False
    # Every call Claude makes must be answered in its own tool result, a skipped one included.
    tool_result_only_continuation = True

    def __init__(
        self,
        *,
        model: str,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        message_fitter: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._effort = reasoning_effort
        self._enable_thinking = enable_thinking
        self._fitter = message_fitter
        self._sdk: Any = None
        self._client: Any = None
        self._pump: Optional[asyncio.Future] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._names: dict[str, str] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._announced: list[tuple[str, str, Any]] = []
        self._calls: dict[str, asyncio.Future] = {}
        self._waiting: list[str] = []
        self._seen: tuple[int, Any] = (0, None)
        self._turn_open = False
        self._tools_open = False
        self._closed = False
        self._ok = False
        self._completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        self._created = int(time.time())

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        cancel_event: threading.Event | None,
    ) -> AsyncIterator[str]:
        return self._step(messages, tools, tool_choice, cancel_event)

    def _chunk(
        self,
        delta: dict[str, Any],
        finish_reason: Optional[str] = None,
    ) -> str:
        return "data: " + json.dumps(
            {
                "id": self._completion_id,
                "object": "chat.completion.chunk",
                "created": self._created,
                "model": self._model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    async def _step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        cancel_event: threading.Event | None,
    ) -> AsyncIterator[str]:
        self._tools_open = bool(tools) and tool_choice != "none"
        fresh = self._fresh(messages)
        self._seen = (len(messages), messages[-1].get("content") if messages else None)
        self._ok = False
        try:
            if self._closed:
                yield _error_sse_line(502, "The Claude Code session has ended.", PROVIDER_TYPE)
                return
            if self._client is None:
                async for line in self._start(messages, tools if self._tools_open else None):
                    yield line
                if not self._turn_open:
                    return
            elif self._waiting:
                self._answer(fresh)
            elif not self._turn_open:
                blocks = []
                for message in fresh:
                    if message.get("role") == "user":
                        blocks.extend(_user_blocks(message.get("content")))
                if not blocks:
                    self._ok = True
                    yield self._chunk({}, "stop")
                    yield _DONE
                    return
                await self._send(blocks)
            async for line in self._read(cancel_event):
                yield line
        finally:
            if not self._ok:
                await self.aclose()

    def _fresh(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        count, last = self._seen
        fresh = list(messages[count:])
        if 0 < count <= len(messages):
            previous = messages[count - 1]
            extra = _appended(last, previous.get("content"))
            if extra and previous.get("role") == "user":
                fresh.insert(0, {"role": "user", "content": extra})
        return fresh

    async def _start(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> AsyncIterator[str]:
        sdk = _sdk()
        if sdk is None:
            yield _error_sse_line(503, SDK_MISSING, PROVIDER_TYPE)
            return
        cli = find_cli()
        if cli is None:
            yield _error_sse_line(503, CLI_MISSING, PROVIDER_TYPE)
            return
        outdated = outdated_message(await cli_version(cli))
        if outdated:
            yield _error_sse_line(503, outdated, PROVIDER_TYPE)
            return
        max_tokens = self._max_tokens
        if self._fitter is not None:
            messages, max_tokens, truncation_line = await self._fitter(messages)
            if truncation_line:
                yield truncation_line
        system, blocks = build_prompt(messages, tools = bool(tools))
        if blocks is None:
            yield _error_sse_line(
                400, "Claude Agent needs a user message to reply to.", PROVIDER_TYPE
            )
            return
        if self._effort in EFFORTS or self._enable_thinking is False:
            await ensure_catalog(sdk, cli)
        extra: dict[str, Any] = {}
        if tools:
            extra["mcp_servers"] = {TOOL_SERVER: self._server(sdk, tools)}
            extra["allowed_tools"] = [f"{TOOL_PREFIX}*"]
            extra["hooks"] = {"PreToolUse": [sdk.HookMatcher(hooks = [self._before])]}
        options = _options(
            sdk,
            cli,
            model = self._model,
            system_prompt = {"type": "preset", "preset": "claude_code", "append": system},
            include_partial_messages = True,
            verbatim_prompts = True,
            thinking = thinking_config(self._model, self._enable_thinking),
            effort = clamp_effort(self._model, self._effort),
            **extra,
        )
        if max_tokens:
            options.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        self._sdk = sdk
        self._client = sdk.ClaudeSDKClient(options)
        try:
            await self._client.connect()
            info = await self._client.get_server_info()
            if isinstance(info, dict):
                remember_models(info)
        except sdk.ClaudeSDKError as exc:
            yield _error_sse_line(502, f"Claude Code did not start: {exc}", PROVIDER_TYPE)
            return
        self._pump = asyncio.ensure_future(self._read_all(self._client))
        await self._send(blocks)

    def _server(self, sdk, tools: list[dict[str, Any]]):
        definitions = []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name:
                continue
            short = tool_name(name)
            schema = dict(function.get("parameters") or {})
            if not isinstance(schema.get("type"), str):
                schema["type"] = "object"
            schema.setdefault("properties", {})
            self._names[TOOL_PREFIX + short] = name
            self._schemas[TOOL_PREFIX + short] = schema
            definitions.append(
                sdk.tool(
                    short,
                    function.get("description") or "",
                    schema,
                    # Not read-only, so Claude Code runs the calls one at a time, in order.
                    annotations = sdk.ToolAnnotations(readOnlyHint = False, maxResultSizeChars = 500_000),
                )(partial(self._handle, TOOL_PREFIX + short))
            )
        return sdk.create_sdk_mcp_server(TOOL_SERVER, tools = definitions)

    async def _before(self, data: dict[str, Any], tool_use_id: Optional[str], context: Any):
        name = data.get("tool_name")
        if name in self._names:
            self._announced.append(
                (data.get("tool_use_id") or tool_use_id, name, data.get("tool_input"))
            )
        return {}

    def _claim(self, name: str, arguments: Any) -> Optional[str]:
        # The MCP handler never sees Claude's call id, so it takes the one the PreToolUse hook announced.
        for exact in (True, False):
            for index, (call_id, announced, given) in enumerate(self._announced):
                if announced == name and (given == arguments or not exact):
                    del self._announced[index]
                    return call_id
        return None

    def _future(self, call_id: str) -> asyncio.Future:
        if call_id not in self._calls:
            self._calls[call_id] = asyncio.get_running_loop().create_future()
        return self._calls[call_id]

    async def _handle(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        call_id = self._claim(name, arguments)
        if call_id is None or self._closed:
            return {"content": [{"type": "text", "text": _TOOLS_CLOSED}], "is_error": True}
        # The handler can run before the reader reaches message_stop; the reader settles the future.
        try:
            return await self._future(call_id)
        finally:
            self._calls.pop(call_id, None)

    def _valid(self, name: str, arguments: Any) -> bool:
        # The SDK answers a call that fails its schema itself, so Studio must not run that one.
        try:
            import jsonschema
        except ImportError:
            return True
        try:
            jsonschema.validate(instance = arguments, schema = self._schemas[name])
        except jsonschema.ValidationError:
            return False
        except Exception:
            return True
        return True

    def _answer(self, fresh: list[dict[str, Any]]) -> None:
        results = {}
        unmatched = []
        extras: list[dict[str, Any]] = []
        for message in fresh:
            role = message.get("role")
            if role == "tool":
                if message.get("tool_call_id") in self._waiting:
                    results[message["tool_call_id"]] = message
                else:
                    unmatched.append(message)
            elif role == "user":
                extras.extend(_mcp_content(message.get("content")))
        for index, call_id in enumerate(self._waiting):
            message = results.get(call_id) or (unmatched.pop(0) if unmatched else None)
            text = _text(message.get("content")) if message else _NOT_RUN
            content = _mcp_content(message.get("content")) if message else []
            content = content or [{"type": "text", "text": text or "(no output)"}]
            if index == len(self._waiting) - 1:
                content = content + extras
            future = self._future(call_id)
            if not future.done():
                future.set_result(
                    {"content": content, "is_error": message is None or _is_error(text)}
                )
        self._waiting = []

    async def _send(self, blocks: list[dict[str, Any]]) -> None:
        async def turn():
            yield {
                "type": "user",
                "message": {"role": "user", "content": blocks},
                "parent_tool_use_id": None,
            }

        self._turn_open = True
        await self._client.query(turn())

    async def _read_all(self, client) -> None:
        try:
            async for message in client.receive_messages():
                self._queue.put_nowait(message)
        except Exception as exc:
            self._queue.put_nowait(_Ended(exc))
        else:
            self._queue.put_nowait(_Ended(None))

    async def _next(self, watcher: Optional[asyncio.Future]) -> Any:
        if watcher is None:
            return await self._queue.get()
        read = asyncio.ensure_future(self._queue.get())
        try:
            await asyncio.wait({read, watcher}, return_when = asyncio.FIRST_COMPLETED)
        finally:
            if not read.done():
                read.cancel()
        return read.result() if read.done() and not read.cancelled() else None

    async def _read(self, cancel_event: threading.Event | None) -> AsyncIterator[str]:
        sdk = self._sdk
        thinking = False
        usage: dict[str, Any] = {}
        error = None
        rate_limit = None
        calls: list[tuple[str, str, Any]] = []
        watcher = (
            asyncio.ensure_future(_await_cancel(cancel_event)) if cancel_event is not None else None
        )
        try:
            while True:
                message = await self._next(watcher)
                if message is None:
                    return
                if isinstance(message, _Ended):
                    if thinking:
                        yield self._chunk({"content": "</think>"})
                    detail = f"Claude Code stopped: {message.error}" if message.error else None
                    yield _error_sse_line(
                        502, detail or "Claude Code ended without a reply.", PROVIDER_TYPE
                    )
                    return
                if isinstance(message, sdk.StreamEvent):
                    if message.parent_tool_use_id is not None:
                        continue
                    event = message.event
                    kind = event.get("type")
                    if kind == "message_start":
                        usage = dict((event.get("message") or {}).get("usage") or {})
                        calls = []
                    elif kind == "message_delta":
                        usage.update(
                            {k: v for k, v in (event.get("usage") or {}).items() if v is not None}
                        )
                    elif kind == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                            text = delta["thinking"]
                            if not thinking:
                                text = f"<think>{text}"
                                thinking = True
                            yield self._chunk({"content": text})
                        elif delta.get("type") == "text_delta" and delta.get("text"):
                            if thinking:
                                yield self._chunk({"content": "</think>"})
                                thinking = False
                            yield self._chunk({"content": delta["text"]})
                    elif kind == "message_stop" and calls:
                        runnable = []
                        for call in calls:
                            if self._tools_open and self._valid(call[1], call[2]):
                                runnable.append(call)
                            elif not self._future(call[0]).done():
                                self._future(call[0]).set_result(
                                    {
                                        "content": [{"type": "text", "text": _TOOLS_CLOSED}],
                                        "is_error": True,
                                    }
                                )
                        calls = []
                        if not runnable:
                            continue
                        if thinking:
                            yield self._chunk({"content": "</think>"})
                            thinking = False
                        for index, (call_id, name, arguments) in enumerate(runnable):
                            self._future(call_id)
                            yield self._chunk(
                                {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": self._names[name],
                                                "arguments": json.dumps(arguments),
                                            },
                                        }
                                    ]
                                }
                            )
                        self._waiting = [call[0] for call in runnable]
                        self._ok = True
                        yield self._chunk({}, "tool_calls")
                        usage_line = _build_usage_chunk(self._completion_id, "anthropic", usage)
                        if usage_line:
                            yield usage_line
                        yield _DONE
                        return
                elif isinstance(message, sdk.AssistantMessage):
                    if message.parent_tool_use_id is not None:
                        continue
                    if message.error:
                        error = message.error
                    for block in message.content or []:
                        if isinstance(block, sdk.ToolUseBlock) and block.name in self._names:
                            calls.append((block.id, block.name, block.input))
                elif isinstance(message, sdk.UserMessage):
                    answered = {
                        block.tool_use_id
                        for block in (message.content if isinstance(message.content, list) else [])
                        if isinstance(block, sdk.ToolResultBlock)
                    }
                    self._announced = [a for a in self._announced if a[0] not in answered]
                elif isinstance(message, sdk.RateLimitEvent):
                    rate_limit = message.rate_limit_info
                elif isinstance(message, sdk.ResultMessage):
                    self._turn_open = False
                    if thinking:
                        yield self._chunk({"content": "</think>"})
                    truncated = error == "max_output_tokens"
                    failed = (
                        message.is_error
                        or message.subtype != "success"
                        or (message.api_error_status or 0) >= 400
                    )
                    if failed and not truncated:
                        status, text = failure(message, error, rate_limit)
                        yield _error_sse_line(status, text, PROVIDER_TYPE)
                        return
                    self._ok = True
                    yield self._chunk(
                        {}, "length" if truncated or message.stop_reason == "max_tokens" else "stop"
                    )
                    usage_line = _build_usage_chunk(self._completion_id, "anthropic", usage)
                    if usage_line:
                        yield usage_line
                    yield _DONE
                    return
        finally:
            if watcher is not None:
                watcher.cancel()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ok = True
        client, self._client = self._client, None
        if client is None:
            return
        try:
            if self._turn_open:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.interrupt(), timeout = 3)
            for future in list(self._calls.values()):
                if not future.done():
                    future.set_result(
                        {"content": [{"type": "text", "text": _CANCELLED}], "is_error": True}
                    )
        finally:
            with contextlib.suppress(Exception):
                await client.disconnect()
            if self._pump is not None:
                self._pump.cancel()


async def close_after(
    lines: AsyncIterator[str], transport: ClaudeAgentTransport
) -> AsyncIterator[str]:
    try:
        async for line in lines:
            yield line
    finally:
        with contextlib.suppress(RuntimeError, GeneratorExit, StopAsyncIteration):
            await lines.aclose()
        await transport.aclose()
