# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fixed-host ChatGPT Codex Responses transport and SSE normalization."""

from __future__ import annotations

import asyncio

from contextlib import asynccontextmanager
import hashlib
import json
import secrets
import threading
import time
from typing import Any, AsyncGenerator, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from core.inference.openai_codex_auth import (
    OPENAI_CODEX_COMPATIBILITY_INSTRUCTIONS,
    OPENAI_CODEX_ORIGINATOR,
    OPENAI_CODEX_RESPONSES_URL,
    OPENAI_CODEX_USER_AGENT,
)
from core.inference.openai_responses_shared import (
    normalize_function_schema,

    responses_function_call,
    responses_function_output,
    response_event_type,
    responses_usage_to_chat,
)


class CodexTransportError(RuntimeError):
    def __init__(self, message: str, *, status: int = 502, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.metadata = metadata or {}


class CodexReauthorizationError(CodexTransportError):
    pass


class CodexQuotaError(CodexTransportError):
    pass




def _codex_call_id(value: Any) -> str | None:
    """Return a stable Responses-compatible call ID (the API caps IDs at 64 chars)."""
    if not isinstance(value, str) or not value:
        return None
    if len(value) <= 64:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"{value[:31]}_{digest}"



def _responses_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Translate chat history to the Responses input shape used by Pi/Codex."""
    instructions = [OPENAI_CODEX_COMPATIBILITY_INSTRUCTIONS]
    items: list[dict[str, Any]] = []
    assistant_index = 0
    for message in messages:
        role, content = message.get("role"), message.get("content", "")
        if role == "system":
            if isinstance(content, str) and content:
                instructions.append(content)
            continue
        if role == "tool":
            call_id = _codex_call_id(message.get("tool_call_id"))
            if call_id:
                output = content if isinstance(content, str) else json.dumps(content)
                items.append(responses_function_output(call_id, output))
            continue
        if role == "assistant":
            extra = message.get("extra_content")
            for item in extra.get("openai_codex_reasoning", []) if isinstance(extra, dict) else []:
                if isinstance(item, dict) and item.get("type") == "reasoning" and isinstance(item.get("encrypted_content"), str):
                    replay = {"type": "reasoning", "encrypted_content": item["encrypted_content"], "summary": item.get("summary", [])}
                    if isinstance(item.get("id"), str):
                        replay["id"] = item["id"]
                    items.append(replay)
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and function.get("name"):
                    arguments = function.get("arguments", "")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    call_id = _codex_call_id(call.get("id"))
                    if call_id:
                        items.append(
                            responses_function_call(
                                call_id, function["name"], arguments
                            )
                        )
            output_parts: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                output_parts.append({"type": "output_text", "text": content, "annotations": []})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        output_parts.append({"type": "output_text", "text": part.get("text", ""), "annotations": []})
            if output_parts:
                items.append({
                    "type": "message",
                    "id": f"msg_unsloth_{assistant_index}",
                    "role": "assistant",
                    "content": output_parts,
                    "status": "completed",
                })
                assistant_index += 1
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            if content:
                items.append({"role": "user", "content": [{"type": "input_text", "text": content}]})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    parts.append({"type": "input_text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        parts.append({"type": "input_image", "detail": "auto", "image_url": url})
            if parts:
                items.append({"role": "user", "content": parts})
    return "\n\n".join(instructions), items


def _chunk(completion_id: str, model: str, delta: dict[str, Any], finish_reason: str | None = None, usage: Any = None) -> str:
    body: dict[str, Any] = {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        body["usage"] = usage
    return "data: " + json.dumps(body, separators=(",", ":"))


def _create_http_client() -> httpx.AsyncClient:
    kwargs = {"timeout": httpx.Timeout(120.0, connect=20.0), "follow_redirects": False}
    try:
        return httpx.AsyncClient(**kwargs)
    except (ImportError, ValueError) as exc:
        if "Unknown scheme for proxy URL" not in str(exc) and "socksio" not in str(exc):
            raise
        return httpx.AsyncClient(**kwargs, trust_env=False)


def _validated_responses_url() -> str:
    parsed = urlparse(OPENAI_CODEX_RESPONSES_URL)
    if parsed.scheme != "https" or parsed.hostname != "chatgpt.com" or parsed.port is not None or parsed.path != "/backend-api/codex/responses" or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise RuntimeError("ChatGPT Codex endpoint configuration is invalid.")
    return OPENAI_CODEX_RESPONSES_URL


def _quota_metadata(response: httpx.Response) -> dict[str, Any]:
    values = {
        "retry_after": response.headers.get("retry-after"),
        "requests_reset": response.headers.get("x-ratelimit-reset-requests"),
        "tokens_reset": response.headers.get("x-ratelimit-reset-tokens"),
    }
    return {key: value for key, value in values.items() if value}


async def _upstream_error_detail(response: httpx.Response) -> str | None:
    """Extract only structured, bounded upstream error text; never echo HTML."""
    try:
        await response.aread()
        payload = response.json()
    except (ValueError, json.JSONDecodeError, httpx.HTTPError, AttributeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code") or error.get("type")
        detail = message if isinstance(message, str) and message.strip() else code
    else:
        detail = error if isinstance(error, str) else None
    if not isinstance(detail, str):
        return None
    return " ".join(detail.split())[:500] or None


async def _wait_for_cancel(cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(0.05)


@asynccontextmanager
async def _stream_response(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    cancel_event: threading.Event | None,
):
    """Open a streaming response while allowing pre-header cancellation."""
    context = client.stream("POST", url, headers=headers, json=body)
    if cancel_event is None:
        async with context as response:
            yield response
        return
    if cancel_event.is_set():
        yield None
        return

    enter_task = asyncio.create_task(context.__aenter__())
    cancel_task = asyncio.create_task(_wait_for_cancel(cancel_event))
    done, _pending = await asyncio.wait(
        {enter_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if cancel_task in done and cancel_event.is_set():
        enter_task.cancel()
        await asyncio.gather(enter_task, return_exceptions=True)
        yield None
        return

    cancel_task.cancel()
    await asyncio.gather(cancel_task, return_exceptions=True)
    response = await enter_task
    try:
        yield response
    finally:
        await context.__aexit__(None, None, None)


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_TRANSIENT_RETRIES = 2


def _is_terminal_quota(detail: str | None) -> bool:
    if not detail:
        return False
    lowered = detail.lower()
    return any(
        marker in lowered
        for marker in (
            "usage limit",
            "insufficient_quota",
            "out of budget",
            "quota exceeded",
            "available balance",
            "billing",
        )
    )


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    raw = response.headers.get("retry-after-ms")
    if raw:
        try:
            return min(60.0, max(0.0, float(raw) / 1000.0))
        except ValueError:
            pass
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return min(60.0, max(0.0, float(raw)))
        except ValueError:
            pass
    return float(2 ** attempt)


async def _retry_pause(delay: float, cancel_event: threading.Event | None) -> None:
    deadline = asyncio.get_running_loop().time() + delay
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.1, remaining))


@asynccontextmanager
async def _validated_stream_response(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    cancel_event: threading.Event | None,
    refresh_access: Callable[[], Awaitable[tuple[str, str]]] | None,
):
    refreshed = False

    token = headers.get("Authorization", "").removeprefix("Bearer ")
    for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
        yielded = False
        try:
            async with _stream_response(
                client,
                url=url,
                headers=headers,
                body=body,
                cancel_event=cancel_event,
            ) as response:
                if response is None:
                    yield None
                    return
                if 200 <= response.status_code < 300:
                    yielded = True
                    yield response
                    return
                if 300 <= response.status_code < 400:
                    raise CodexTransportError(
                        "ChatGPT Codex endpoint returned a forbidden redirect."
                    )
                detail = await _upstream_error_detail(response)
                if response.status_code == 401 and refresh_access is not None and not refreshed:
                    try:
                        token, account_id = await refresh_access()
                    except Exception as exc:
                        raise CodexReauthorizationError(
                            "ChatGPT authorization expired. Reconnect this connection.",
                            status=401,
                            metadata={"access_token": token},
                        ) from exc
                    headers["Authorization"] = f"Bearer {token}"
                    headers["chatgpt-account-id"] = account_id
                    refreshed = True
                    continue
                if response.status_code == 401:
                    raise CodexReauthorizationError(
                        "ChatGPT authorization expired. Reconnect this connection.",
                        status=401,
                        metadata={"access_token": token},
                    )
                retryable = (
                    response.status_code in _RETRYABLE_STATUSES
                    and not _is_terminal_quota(detail)
                )
                if retryable and attempt < _MAX_TRANSIENT_RETRIES:
                    await _retry_pause(_retry_delay_seconds(response, attempt), cancel_event)
                    if cancel_event is not None and cancel_event.is_set():
                        yield None
                        return
                    continue
                if response.status_code == 429:
                    raise CodexQuotaError(
                        "ChatGPT subscription quota is temporarily unavailable.",
                        status=429,
                        metadata=_quota_metadata(response),
                    )
                suffix = f" {detail}" if detail else ""
                raise CodexTransportError(
                    f"ChatGPT Codex request failed ({response.status_code}).{suffix}",
                    status=response.status_code,
                )
        except httpx.HTTPError as exc:
            if yielded:
                raise
            if attempt >= _MAX_TRANSIENT_RETRIES:
                raise CodexTransportError("Could not reach ChatGPT Codex.") from exc
            await _retry_pause(float(2 ** attempt), cancel_event)
            if cancel_event is not None and cancel_event.is_set():
                yield None
                return
    raise CodexTransportError("Could not reach ChatGPT Codex.")


class OpenAICodexClient:
    def __init__(
        self,
        access_token: str,
        account_id: str,
        *,
        refresh_access: Callable[[], Awaitable[tuple[str, str]]] | None = None,
    ) -> None:
        self._token, self._account_id = access_token, account_id
        self._refresh_access = refresh_access
        self._client = _create_http_client()


    async def _refresh_credentials(self) -> tuple[str, str]:
        if self._refresh_access is None:
            raise CodexReauthorizationError(
                "ChatGPT authorization expired. Reconnect this connection.",
                status=401,
            )
        token, account_id = await self._refresh_access()
        self._token, self._account_id = token, account_id
        return token, account_id

    async def close(self) -> None:
        self._token = ""
        await self._client.aclose()

    async def stream(self, *, provider_id: str, thread_id: str | None, messages: list[dict[str, Any]], model: str, max_tokens: int | None, reasoning_effort: str | None, tools: list[dict[str, Any]] | None, tool_choice: Any, cancel_event: threading.Event | None = None) -> AsyncGenerator[str, None]:
        instructions, input_items = _responses_input(messages)
        conversation_id = thread_id or secrets.token_urlsafe(24)
        affinity = hashlib.sha256(
            f"{provider_id}\0{self._account_id}\0{conversation_id}".encode()
        ).hexdigest()[:48]
        body: dict[str, Any] = {"model": model, "instructions": instructions, "input": input_items, "store": False, "stream": True, "text": {"verbosity": "low"}, "include": ["reasoning.encrypted_content"], "prompt_cache_key": affinity, "tool_choice": "auto", "parallel_tool_calls": True}
        # Pi intentionally does not forward a token cap here. ChatGPT's Codex
        # Responses endpoint rejects max_output_tokens even though the public
        # Responses API accepts it; the subscription service applies its own cap.
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        if tools:
            converted = []
            for tool in tools:
                function = tool.get("function") if isinstance(tool, dict) and tool.get("type") == "function" else None
                if isinstance(function, dict) and function.get("name"):
                    converted.append({"type": "function", "name": function["name"], "description": function.get("description", ""), "parameters": normalize_function_schema(function.get("parameters"))})
            if converted:
                body["tools"] = converted
        if isinstance(tool_choice, str) and tool_choice in ("auto", "none", "required"):
            body["tool_choice"] = tool_choice
        elif isinstance(tool_choice, dict):
            fn = tool_choice.get("function") or {}
            if fn.get("name"):
                body["tool_choice"] = {"type": "function", "name": fn["name"]}
        request_id = hashlib.sha256(f"{affinity}\0{time.time_ns()}".encode()).hexdigest()[:32]
        headers = {"Authorization": f"Bearer {self._token}", "chatgpt-account-id": self._account_id, "originator": OPENAI_CODEX_ORIGINATOR, "User-Agent": OPENAI_CODEX_USER_AGENT, "OpenAI-Beta": "responses=experimental", "Accept": "text/event-stream", "Content-Type": "application/json", "session-id": affinity, "x-client-request-id": affinity}
        completion_id = f"chatcmpl-codex-{request_id}"
        emitted_terminal, saw_tool_call = False, False
        reasoning_items: list[dict[str, Any]] = []
        cancel_task: asyncio.Task | None = None
        try:
            async with _validated_stream_response(
                self._client,
                url=_validated_responses_url(),
                headers=headers,
                body=body,
                cancel_event=cancel_event,
                refresh_access=(
                    self._refresh_credentials if self._refresh_access is not None else None
                ),
            ) as response:
                if response is None:
                    return
                if cancel_event is not None:
                    async def _close_on_cancel() -> None:
                        await _wait_for_cancel(cancel_event)
                        await response.aclose()

                    cancel_task = asyncio.create_task(_close_on_cancel())
                event_name, tool_indexes = "", {}
                try:
                    async for line in response.aiter_lines():
                        if cancel_event is not None and cancel_event.is_set():
                            return
                        if not line:
                            event_name = ""
                            continue
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            event = json.loads(raw)
                            kind = response_event_type(event, event_name)
                        except (ValueError, json.JSONDecodeError) as exc:
                            raise CodexTransportError("ChatGPT returned a malformed stream.") from exc
                        if kind in ("response.created", "response.in_progress"):
                            continue
                        if kind in ("response.output_text.delta", "response.refusal.delta"):
                            delta = event.get("delta")
                            if not isinstance(delta, str):
                                raise CodexTransportError("ChatGPT returned a malformed stream.")
                            yield _chunk(completion_id, model, {"content": delta})
                        elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                            delta = event.get("delta")
                            if isinstance(delta, str):
                                yield _chunk(completion_id, model, {"reasoning_content": delta})
                        elif kind == "response.output_item.added":
                            item = event.get("item")
                            if isinstance(item, dict) and item.get("type") == "function_call":
                                call_id, name = item.get("call_id") or item.get("id"), item.get("name")
                                if isinstance(call_id, str) and isinstance(name, str):
                                    saw_tool_call = True
                                    index = len(set(tool_indexes.values()))
                                    tool_indexes[call_id] = index
                                    if isinstance(item.get("id"), str):
                                        tool_indexes[item["id"]] = index
                                    yield _chunk(completion_id, model, {"tool_calls": [{"index": index, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}]})
                        elif kind == "response.function_call_arguments.delta":
                            call_id, delta = event.get("call_id") or event.get("item_id"), event.get("delta")
                            if not isinstance(call_id, str) or not isinstance(delta, str):
                                raise CodexTransportError("ChatGPT returned a malformed stream.")
                            saw_tool_call = True
                            index = tool_indexes.setdefault(call_id, len(set(tool_indexes.values())))
                            yield _chunk(completion_id, model, {"tool_calls": [{"index": index, "function": {"arguments": delta}}]})
                        elif kind == "response.output_item.done":
                            item = event.get("item") or {}
                            if isinstance(item, dict) and item.get("type") == "function_call":
                                call_id = item.get("call_id") or item.get("id")
                                if isinstance(call_id, str) and call_id not in tool_indexes and isinstance(item.get("name"), str):
                                    saw_tool_call = True
                                    index = tool_indexes.setdefault(call_id, len(set(tool_indexes.values())))
                                    yield _chunk(completion_id, model, {"tool_calls": [{"index": index, "id": call_id, "type": "function", "function": {"name": item["name"], "arguments": item.get("arguments", "")}}]})
                            elif isinstance(item, dict) and item.get("type") == "reasoning" and isinstance(item.get("encrypted_content"), str):
                                reasoning_items.append({"type": "reasoning", "id": item.get("id"), "encrypted_content": item["encrypted_content"], "summary": item.get("summary", [])})
                        elif kind in ("response.completed", "response.incomplete"):
                            response_body = event.get("response") or {}
                            delta: dict[str, Any] = {}
                            if reasoning_items:
                                delta["extra_content"] = {"openai_codex_reasoning": reasoning_items}
                            finish = "length" if kind == "response.incomplete" else ("tool_calls" if saw_tool_call else "stop")
                            yield _chunk(completion_id, model, delta, finish, responses_usage_to_chat(response_body.get("usage")))
                            emitted_terminal = True
                            break
                        elif kind in ("response.failed", "error"):
                            error = event.get("error") or (event.get("response") or {}).get("error") or {}
                            code = error.get("code") if isinstance(error, dict) else None
                            raise CodexTransportError(f"ChatGPT Codex generation failed{f' ({code})' if code else ''}.")
                except httpx.HTTPError:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    raise
        finally:
            if cancel_task is not None:
                cancel_task.cancel()
        if not emitted_terminal and not (cancel_event is not None and cancel_event.is_set()):
            raise CodexTransportError("ChatGPT stream ended before completion.")
