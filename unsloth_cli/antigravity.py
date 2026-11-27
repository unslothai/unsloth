# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Session-scoped Gemini streaming endpoint for the Antigravity CLI."""

import json
import secrets
import selectors
import socket
import threading
import urllib.error
import urllib.request
from collections import defaultdict, deque
from contextlib import contextmanager, suppress
from http.client import IncompleteRead
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from unsloth_cli._inference import _USER_AGENT, urlopen_no_redirect


# agy accepts values beginning with a dash, including a literal "--settings"
# as the prompt. Only inspect option positions when collecting its override.
_VALUE_FLAGS = frozenset(
    "p print prompt i prompt-interactive model agent project conversation r resume "
    "effort mode add-dir log-file input-format output-format json-schema print-timeout".split()
)


def session_settings(args):
    """Extract agy's last --settings, preserving its other arguments and their values."""
    remaining = []
    value = None
    args = iter(args)
    for arg in args:
        if arg == "--":
            remaining.extend([arg, *args])
            break
        flag, equals, inline = arg.partition("=")
        name = flag.lstrip("-") if flag.startswith("-") else ""
        if name == "settings":
            value = inline if equals else next(args, None)
            if not value:
                raise ValueError("--settings requires a JSON object or a file path.")
        else:
            remaining.append(arg)
            if name in _VALUE_FLAGS and not equals:
                argument = next(args, None)
                if argument is not None:
                    remaining.append(argument)
    if value is None:
        return {}, remaining
    try:
        raw = (
            value
            if value.lstrip().startswith("{")
            else Path(value).expanduser().read_text(encoding = "utf-8-sig")
        )
        settings = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read Antigravity --settings: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError("Antigravity --settings must contain a JSON object.")
    return settings, remaining


def _content_part(part):
    if "text" in part:
        return {"type": "text", "text": part["text"]}
    image = part.get("inlineData", {})
    if image.get("mimeType", "").startswith("image/"):
        url = f"data:{image['mimeType']};base64,{image['data']}"
        return {"type": "image_url", "image_url": {"url": url}}
    raise ValueError("Antigravity sent unsupported content. Use text or an inline image.")


def _chat_request(
    body,
    model,
    overrides,
    *,
    supports_tools = True,
):
    messages = []
    system = body.get("systemInstruction", {}).get("parts", [])
    if system:
        messages.append({"role": "system", "content": "\n".join(p["text"] for p in system)})
    pending = defaultdict(deque)
    tool_media = []
    call_number = 0
    for turn in body["contents"]:
        role = {"model": "assistant", "user": "user"}[turn.get("role", "user")]
        content, calls = [], []
        for part in turn["parts"]:
            if part.get("thought"):
                continue
            if "functionCall" in part:
                call = part["functionCall"]
                call_number += 1
                call_id = call.get("id") or f"call_{call_number}"
                pending[call["name"]].append(call_id)
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                )
            elif "functionResponse" in part:
                result = part["functionResponse"]
                ids = pending[result["name"]]
                if not ids:
                    raise ValueError("Tool result has no matching function call.")
                call_id = result.get("id") or ids[0]
                ids.remove(call_id)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result["response"], ensure_ascii = False),
                    }
                )
                # agy puts functionResponse in a model turn. Tool images are
                # inputs from the user/tool side in OpenAI vision templates.
                tool_media.extend(_content_part(p) for p in result.get("parts", []))
                if not ids:
                    del pending[result["name"]]
                if tool_media and not pending:
                    messages.append({"role": "user", "content": tool_media})
                    tool_media = []
            else:
                content.append(_content_part(part))
        if content or calls:
            if all(p["type"] == "text" for p in content):
                content = "\n".join(p["text"] for p in content)
            message = {"role": role, "content": content}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
    request = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # agy includes declarations even when the custom model disables tools.
    if supports_tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": f.get(
                        "parametersJsonSchema", {"type": "object", "properties": {}}
                    ),
                },
            }
            for group in body.get("tools", [])
            for f in group["functionDeclarations"]
        ]
        if tools:
            request["tools"] = tools
    config = body.get("generationConfig", {})
    for name, field in (
        ("temperature", "temperature"),
        ("topP", "top_p"),
        ("topK", "top_k"),
        ("stopSequences", "stop"),
    ):
        if name in config:
            request[field] = config[name]
    if config.get("maxOutputTokens"):
        request["max_tokens"] = config["maxOutputTokens"]
    level = (config.get("thinkingConfig") or {}).get("thinkingLevel")
    if overrides.get("enable_thinking") is not False and level in (
        "MINIMAL",
        "LOW",
        "MEDIUM",
        "HIGH",
        "XHIGH",
        "MAX",
    ):
        request["reasoning_effort"] = level.lower()
    request.update(overrides)
    if "max_tokens" in overrides and config.get("maxOutputTokens"):
        request["max_tokens"] = min(overrides["max_tokens"], config["maxOutputTokens"])
    return request


def _candidate(parts, finish = None):
    candidate = {"index": 0, "content": {"role": "model", "parts": parts}}
    if finish:
        candidate["finishReason"] = finish
    return {"candidates": [candidate]}


def _gemini_stream(lines):
    calls = {}
    usage = {}
    finish = None
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        if "error" in chunk:
            raise ValueError(chunk["error"].get("message", "Unsloth inference failed"))
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("reasoning_content"):
                yield _candidate([{"text": delta["reasoning_content"], "thought": True}])
            if delta.get("content"):
                yield _candidate([{"text": delta["content"]}])
            for call in delta.get("tool_calls", []):
                slot = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function", {})
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    if not finish:
        raise ValueError("Unsloth closed the response before inference finished.")
    parts = []
    if finish in ("stop", "tool_calls"):
        for _, call in sorted(calls.items()):
            function = {"name": call["name"], "args": json.loads(call["arguments"] or "{}")}
            if call.get("id"):
                function["id"] = call["id"]
            parts.append({"functionCall": function})
    elif finish not in ("length", "content_filter"):
        raise ValueError(f"Unsloth inference ended with {finish!r}.")
    last = _candidate(
        parts, {"length": "MAX_TOKENS", "content_filter": "SAFETY"}.get(finish, "STOP")
    )
    if usage:
        last["usageMetadata"] = {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        }
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is not None:
            last["usageMetadata"]["cachedContentTokenCount"] = cached
    yield last


@contextmanager
def _cancel_on_disconnect(client):
    done, cancelled = threading.Event(), threading.Event()
    upstream = None

    def shutdown_upstream():
        if upstream is not None:
            with suppress(OSError):
                # Wake the reader without clearing an SSLSocket's TLS state from this thread.
                socket.socket.shutdown(upstream, socket.SHUT_RDWR)

    def connected(connection):
        nonlocal upstream
        upstream = connection
        if cancelled.is_set():
            shutdown_upstream()

    def watch():
        with selectors.DefaultSelector() as selector:
            selector.register(client, selectors.EVENT_READ)
            while not done.is_set():
                if not selector.select(0.1):
                    continue
                try:
                    if client.recv(1, socket.MSG_PEEK):
                        done.wait(0.1)
                        continue
                except OSError:
                    pass
                cancelled.set()
                shutdown_upstream()
                return

    watcher = threading.Thread(target = watch, daemon = True)
    watcher.start()
    try:
        yield connected, cancelled
    finally:
        done.set()
        watcher.join()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Prompts and tool results belong to the agent session.

    def _error(self, status, message):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": {"code": status, "message": message}}).encode())

    def do_POST(self):
        if unquote(urlsplit(self.path).path) not in self.server.model_paths:
            return self._error(
                404, "Select the Unsloth model to use this local Antigravity endpoint."
            )
        if not secrets.compare_digest(
            self.headers.get("x-goog-api-key", "").encode(), self.server.token.encode()
        ):
            return self._error(401, "Invalid session key")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64 * 1024 * 1024:
                return self._error(413, "Invalid request size")
            body = _chat_request(
                json.loads(self.rfile.read(length)),
                self.server.model,
                self.server.overrides,
                supports_tools = self.server.supports_tools,
            )
        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            return self._error(400, str(exc))
        request = urllib.request.Request(
            self.server.url,
            json.dumps(body).encode(),
            headers = {
                "Authorization": f"Bearer {self.server.key}",
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
        )
        with _cancel_on_disconnect(self.connection) as (connected, cancelled):
            with suppress(BrokenPipeError, ConnectionResetError):
                self._forward(request, connected, cancelled)

    def _forward(self, request, connected, cancelled):
        try:
            response = urlopen_no_redirect(request, timeout = 1200, on_connect = connected)
        except urllib.error.HTTPError as exc:
            with exc:
                if not cancelled.is_set():
                    self._error(exc.code, exc.read(4096).decode(errors = "replace"))
            return
        except OSError as exc:
            if not cancelled.is_set():
                self._error(502, str(exc))
            return

        with response:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for event in _gemini_stream(response):
                    self._send_event(event)
            except (BrokenPipeError, ConnectionResetError):
                return
            except (IncompleteRead, OSError, ValueError, KeyError, TypeError) as exc:
                if not cancelled.is_set():
                    self._send_event({"error": {"code": 502, "message": str(exc)}})

    def _send_event(self, event):
        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.flush()


@contextmanager
def antigravity_endpoint(
    base,
    key,
    model,
    overrides,
    *,
    supports_tools = True,
):
    with ThreadingHTTPServer(("127.0.0.1", 0), _Handler) as server:
        address = f"127.0.0.1:{server.server_port}"
        server.model_paths = {
            f"/v1beta/models/{name}:streamGenerateContent"
            for name in ("unsloth", f"{address}/models/unsloth")
        }
        server.url = f"{base.rstrip('/')}/v1/chat/completions"
        server.key = key
        server.model = model
        server.overrides = overrides
        server.supports_tools = supports_tools
        server.token = secrets.token_urlsafe(32)
        thread = threading.Thread(target = server.serve_forever, daemon = True)
        thread.start()
        try:
            yield f"http://{address}", server.token
        finally:
            server.shutdown()
            thread.join()
