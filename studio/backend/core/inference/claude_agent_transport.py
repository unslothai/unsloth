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

from core.inference import claude_agent_sessions as sessions
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
    catch_up,
    clamp_effort,
    cli_version,
    ensure_catalog,
    failure,
    find_cli,
    image_url,
    outdated_message,
    remember_models,
    split_prompt,
    thinking_config,
    workdir,
)
from core.inference.external_provider import _build_usage_chunk, _error_sse_line
from loggers import get_logger

logger = get_logger(__name__)

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


class _Retry(Exception):
    pass


def _fork(sdk, record: dict[str, Any], keep: int) -> Optional[tuple[str, list[dict[str, Any]]]]:
    # The SDK gives the copy fresh uuids; check the copied prefix before mapping checkpoints onto it.
    source_id, kept = record["claude_session_id"], record["checkpoints"][:keep]
    directory = str(workdir())
    try:
        source = sdk.get_session_messages(source_id, directory = directory)
        end = [m.uuid for m in source].index(kept[-1]["uuid"])
        forked = sdk.fork_session(
            source_id, directory = directory, up_to_message_id = kept[-1]["uuid"]
        ).session_id
    except Exception:
        return None
    try:
        copied = sdk.get_session_messages(forked, directory = directory)
        if len(copied) != end + 1 or any(
            a.type != b.type or a.message != b.message for a, b in zip(source, copied)
        ):
            raise ValueError("the fork changed the copied conversation")
    except Exception:
        with contextlib.suppress(Exception):
            sdk.delete_session(forked, directory = directory)
        return None
    uuids = {a.uuid: b.uuid for a, b in zip(source, copied)}
    return forked, [
        {"hash": c["hash"], **({"uuid": uuids[c["uuid"]]} if c.get("uuid") in uuids else {})}
        for c in kept
    ]


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
        thread_id: Optional[str] = None,
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
        # Only a chat thread keeps a Claude session; a request without one runs in a throwaway process.
        self._thread = str(thread_id) if thread_id else None
        self._claimed = False
        self._launch: dict[str, Any] = {}
        self._output_cap: Optional[str] = None
        self._tool_defs: list[dict[str, Any]] | None = None
        self._cli: Optional[str] = None
        self._cli_version: Optional[str] = None
        self._previous: Optional[str] = None
        self._session: Optional[str] = None
        self._forks: set[str] = set()
        self._base: list[dict[str, Any]] = []
        self._user: Optional[str] = None
        self._fallback: Optional[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = None
        self._closing: Optional[asyncio.Future] = None
        self._reply: list[str] = []
        self._reply_tools: list[str] = []
        self._reply_uuid: Optional[str] = None
        self._sent = False
        self._completed = False
        self._emitted = False

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
                # Image trimming can delete earlier turns, so take this round's results after its reply.
                reply = max(
                    (i for i, m in enumerate(messages) if m.get("role") == "assistant"), default = -1
                )
                self._answer(messages[reply + 1 :])
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
            while True:
                try:
                    async for line in self._read(cancel_event):
                        yield line
                    break
                except _Retry:
                    # A saved session that no longer resumes starts over with the whole history.
                    error = await self._restart()
                    if error:
                        yield _error_sse_line(502, error, PROVIDER_TYPE)
                        return
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
        version = await cli_version(cli)
        outdated = outdated_message(version)
        if outdated:
            yield _error_sse_line(503, outdated, PROVIDER_TYPE)
            return
        max_tokens = self._max_tokens
        if self._fitter is not None:
            messages, fitted, truncation_line = await self._fitter(messages)
            if max_tokens and fitted:
                max_tokens = min(max_tokens, fitted)
            if truncation_line:
                yield truncation_line
        system, prior, blocks = split_prompt(messages, tools = bool(tools))
        if blocks is None:
            yield _error_sse_line(
                400, "Claude Agent needs a user message to reply to.", PROVIDER_TYPE
            )
            return
        if self._effort in EFFORTS or self._enable_thinking is False:
            await ensure_catalog(sdk, cli)
        self._sdk, self._cli, self._cli_version = sdk, cli, version
        self._tool_defs = tools
        self._launch = {
            "model": self._model,
            "system_prompt": {"type": "preset", "preset": "claude_code", "append": system},
            "include_partial_messages": True,
            "verbatim_prompts": True,
            "thinking": thinking_config(self._model, self._enable_thinking),
            "effort": clamp_effort(self._model, self._effort),
        }
        self._output_cap = str(max_tokens) if max_tokens else None
        full = catch_up(prior) + blocks
        resume, send = None, full
        newest = messages[-1] if messages else {}
        if self._thread and newest.get("role") == "user" and sessions.claim(self._thread):
            self._claimed = True
            self._user = sessions.user_hash(newest)
            fresh_base = [{"hash": group[2]} for group in sessions.groups(prior)]
            resume, self._base, first = await self._plan(sdk, prior, fresh_base)
            if resume is not None:
                send = catch_up(prior[first:]) + blocks
                self._fallback = (full, fresh_base)
        error = await self._open(send, resume)
        if error and self._fallback is not None:
            error = await self._restart()
        if error:
            yield _error_sse_line(502, error, PROVIDER_TYPE)

    async def _plan(
        self, sdk, prior: list[dict[str, Any]], fresh_base: list[dict[str, Any]]
    ) -> tuple[Optional[str], list[dict[str, Any]], int]:
        record = await asyncio.to_thread(sessions.load, self._thread)
        self._previous = record["claude_session_id"] if record else None
        found = sessions.groups(prior)
        mode, keep, aligned = sessions.decide(record, [group[2] for group in found])
        unseen = fresh_base[aligned:]
        start = aligned
        # After a stop, Claude keeps the partial reply it wrote, whatever Studio saved.
        if record and record.get("interrupted") and mode == "resume":
            if start < len(found) and found[start][0] == "assistant":
                start += 1
        first = found[start][1] if start < len(found) else len(prior)
        directory = str(workdir())
        if mode == "resume":
            try:
                exists = await asyncio.to_thread(
                    sdk.get_session_info, self._previous, directory = directory
                )
            except Exception:
                exists = True
            if exists is not None:
                return self._previous, record["checkpoints"][:keep] + unseen, first
        elif mode == "fork":
            forked = await asyncio.to_thread(_fork, sdk, record, keep)
            if forked is not None:
                self._forks.add(forked[0])
                return forked[0], forked[1] + unseen, first
        return None, fresh_base, 0

    async def _open(self, blocks: list[dict[str, Any]], resume: Optional[str]) -> Optional[str]:
        sdk = self._sdk
        launch = dict(self._launch)
        tools = self._tool_defs
        if tools:
            launch["mcp_servers"] = {TOOL_SERVER: self._server(sdk, tools)}
            launch["allowed_tools"] = [f"{TOOL_PREFIX}*"]
            launch["hooks"] = {"PreToolUse": [sdk.HookMatcher(hooks = [self._before])]}
        if self._claimed:
            self._session = resume or str(uuid.uuid4())
            launch["resume" if resume else "session_id"] = self._session
        options = _options(sdk, self._cli, persist = self._claimed, **launch)
        if self._output_cap:
            options.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = self._output_cap
        self._client = sdk.ClaudeSDKClient(options)
        try:
            await self._client.connect()
            info = await self._client.get_server_info()
            if isinstance(info, dict):
                remember_models(info)
        except sdk.ClaudeSDKError as exc:
            await self._drop_client()
            return f"Claude Code did not start: {exc}"
        self._pump = asyncio.ensure_future(self._read_all(self._client))
        await self._send(blocks)
        return None

    async def _drop_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.disconnect()
        if self._pump is not None:
            self._pump.cancel()
            self._pump = None
        self._queue = asyncio.Queue()
        self._turn_open = False

    async def _restart(self) -> Optional[str]:
        blocks, self._base = self._fallback
        self._fallback = None
        await self._drop_client()
        self._announced, self._calls, self._waiting = [], {}, []
        self._reply, self._reply_tools, self._reply_uuid = [], [], None
        return await self._open(blocks, None)

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
        self._sent = True
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
                    if self._fallback and not self._emitted:
                        raise _Retry()
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
                            self._emitted = True
                            yield self._chunk({"content": text})
                        elif delta.get("type") == "text_delta" and delta.get("text"):
                            if thinking:
                                yield self._chunk({"content": "</think>"})
                                thinking = False
                            self._emitted = True
                            self._reply.append(delta["text"])
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
                        self._emitted = True
                        for index, (call_id, name, arguments) in enumerate(runnable):
                            self._future(call_id)
                            self._reply_tools.append(self._names[name])
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
                    if getattr(message, "uuid", None):
                        self._reply_uuid = message.uuid
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
                        if (
                            self._fallback
                            and not self._emitted
                            and status not in (401, 402, 429)
                            and status < 500
                        ):
                            raise _Retry()
                        yield _error_sse_line(status, text, PROVIDER_TYPE)
                        return
                    self._ok = True
                    self._completed = True
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

    async def _drained(self) -> None:
        while not isinstance(await self._queue.get(), (_Ended, self._sdk.ResultMessage)):
            pass

    async def aclose(self) -> None:
        # A client that leaves mid-close cancels the caller; the session record must still be saved.
        if self._closing is None:
            self._closing = asyncio.ensure_future(self._close())
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ok = True
        client, self._client = self._client, None
        try:
            if client is not None:
                try:
                    stopped = self._turn_open
                    if stopped:
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(client.interrupt(), timeout = 3)
                    for future in list(self._calls.values()):
                        if not future.done():
                            future.set_result(
                                {
                                    "content": [{"type": "text", "text": _CANCELLED}],
                                    "is_error": True,
                                }
                            )
                    if stopped and self._claimed:
                        # Let Claude Code write the stopped turn before it exits.
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(self._drained(), timeout = 3)
                finally:
                    with contextlib.suppress(Exception):
                        await client.disconnect()
                    if self._pump is not None:
                        self._pump.cancel()
            if self._claimed:
                await self._save()
        finally:
            if self._claimed:
                sessions.release(self._thread)

    async def _save(self) -> None:
        directory = str(workdir())
        kept = self._session if self._sent else self._previous
        try:
            if self._sent and self._session is not None:
                checkpoints = self._base + [{"hash": self._user}]
                text = "".join(self._reply)
                if self._completed and (sessions.visible(text) or self._reply_tools):
                    reply = {"hash": sessions.reply_hash(text, self._reply_tools)}
                    if self._reply_uuid:
                        reply["uuid"] = self._reply_uuid
                    checkpoints.append(reply)
                record = {
                    "claude_session_id": self._session,
                    "cli_version": self._cli_version,
                    "model": self._model,
                    "checkpoints": checkpoints,
                    "interrupted": not self._completed,
                }
                await asyncio.to_thread(sessions.save, self._thread, record)
            for stale in {self._previous, *self._forks} - {kept, None}:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self._sdk.delete_session, stale, directory = directory)
        except Exception as exc:
            logger.warning("Could not save the Claude Agent session for this chat: %s", exc)


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
