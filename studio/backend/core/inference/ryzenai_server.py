# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-compatible chat server for one RyzenAI NPU model, run in Studio's own Python.

python -m core.inference.ryzenai_server --dll-dir D --model-dir M --model-id I --port P
The bearer key comes from UNSLOTH_RYZENAI_API_KEY. Standard library only, plus jinja2.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

API_KEY_ENV = "UNSLOTH_RYZENAI_API_KEY"


def load_chat_template(model_dir: Path) -> tuple[str, dict[str, str]]:
    """The model's Jinja chat template and its special tokens."""
    config: dict[str, Any] = {}
    try:
        config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        pass
    template = config.get("chat_template")
    jinja = model_dir / "chat_template.jinja"
    if jinja.is_file():
        template = jinja.read_text(encoding = "utf-8")
    if isinstance(template, list):  # [{"name": "default", "template": ...}, ...]
        template = next((t.get("template") for t in template if t.get("name") == "default"), None)
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"{model_dir} has no chat template.")
    tokens = {}
    for key in ("bos_token", "eos_token"):
        value = config.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        tokens[key] = value if isinstance(value, str) else ""
    return template, tokens


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text") or ""
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return ""


def render_prompt(template: str, tokens: dict[str, str], messages: list[dict[str, Any]]) -> str:
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    plain = [
        {"role": str(m.get("role") or "user"), "content": message_text(m.get("content"))}
        for m in messages
        if isinstance(m, dict)
    ]
    return env.from_string(template).render(messages = plain, add_generation_prompt = True, **tokens)


class ThinkSplitter:
    """Moves <think>...</think> text to reasoning_content, as Studio reads it from other servers."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, prompt: str) -> None:
        # R1-style templates open the think block in the prompt itself.
        self.thinking = prompt.rstrip().endswith(self.OPEN)
        self.started = self.thinking
        self.buffer = ""
        self.trim = False  # drop the newlines after </think>

    def feed(self, piece: str) -> tuple[str, str]:
        self.buffer += piece
        reasoning = content = ""
        while True:
            if not self.started and self.buffer.lstrip().startswith(self.OPEN):
                self.buffer = self.buffer.lstrip()[len(self.OPEN) :]
                self.thinking = self.started = True
            tag = self.CLOSE if self.thinking else None
            if tag and tag in self.buffer:
                head, self.buffer = self.buffer.split(tag, 1)
                reasoning += head
                self.thinking, self.trim = False, True
                continue
            # Hold back a possible partial tag; release the rest.
            hold = self._partial(self.buffer, tag or (self.OPEN if not self.started else ""))
            ready, self.buffer = (
                self.buffer[: len(self.buffer) - hold],
                self.buffer[len(self.buffer) - hold :],
            )
            if self.thinking:
                reasoning += ready
            else:
                if self.trim:
                    ready = ready.lstrip("\n")
                    self.trim = not ready
                if ready.strip():
                    self.started = True
                content += ready
            return reasoning, content

    def flush(self) -> tuple[str, str]:
        rest, self.buffer = self.buffer, ""
        return (rest, "") if self.thinking else ("", rest)

    @staticmethod
    def _partial(text: str, tag: str) -> int:
        for size in range(min(len(tag) - 1, len(text)), 0, -1):
            if tag.startswith(text[-size:]):
                return size
        return 0


def search_options(body: dict[str, Any]) -> dict[str, Any]:
    temperature = body.get("temperature")
    sample = isinstance(temperature, (int, float)) and temperature > 0
    options: dict[str, Any] = {"do_sample": sample}
    if sample:
        options["temperature"] = float(temperature)
        for key in ("top_p", "top_k"):
            if isinstance(body.get(key), (int, float)) and body[key] > 0:
                options[key] = body[key]
    return options


class State:
    def __init__(self, model, model_id: str, context_length: int, template, tokens) -> None:
        self.model = model
        self.model_id = model_id
        self.context_length = context_length
        self.template = template
        self.tokens = tokens
        self.lock = threading.Lock()  # one generation at a time on the NPU


def make_handler(state: State, api_key: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args) -> None:  # quiet: Studio logs the chat itself
            pass

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": {"message": message, "type": "invalid_request_error"}})

        def _authorized(self) -> bool:
            sent = self.headers.get("Authorization", "")
            if hmac.compare_digest(sent.encode(), f"Bearer {api_key}".encode()):
                return True
            self._error(401, "Invalid API key.")
            return False

        def do_GET(self) -> None:
            if not self._authorized():
                return
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/health", "/v1/health"):
                self._json(
                    200,
                    {
                        "status": "ok",
                        "model": state.model_id,
                        "context_length": state.context_length,
                    },
                )
            elif path == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": state.model_id, "object": "model", "owned_by": "amd"}],
                    },
                )
            else:
                self._error(404, f"No route {path}.")

        def do_POST(self) -> None:
            if not self._authorized():
                return
            if self.path.split("?", 1)[0].rstrip("/") != "/v1/chat/completions":
                self._error(404, f"No route {self.path}.")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                    raise ValueError("messages must be a list.")
                prompt = render_prompt(state.template, state.tokens, body["messages"])
            except Exception as exc:  # noqa: BLE001 -- a bad request, not a server fault
                self._error(400, f"Invalid request: {exc}")
                return
            with state.lock:
                self._complete(body, prompt)

        def _complete(self, body: dict[str, Any], prompt: str) -> None:
            prompt_tokens = state.model.count_tokens(prompt)
            room = state.context_length - prompt_tokens
            if room <= 0:
                self._error(
                    400,
                    f"The prompt is {prompt_tokens} tokens; this NPU model holds {state.context_length}.",
                )
                return
            asked = body.get("max_completion_tokens") or body.get("max_tokens")
            budget = min(int(asked), room) if isinstance(asked, int) and asked > 0 else room
            stops = body.get("stop") or []
            stops = (
                [stops]
                if isinstance(stops, str)
                else [s for s in stops if isinstance(s, str) and s]
            )
            cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
            stream = bool(body.get("stream"))
            usage_wanted = bool((body.get("stream_options") or {}).get("include_usage"))

            def chunk(delta: dict[str, Any], finish: Optional[str]) -> bytes:
                payload = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                return f"data: {json.dumps(payload)}\n\n".encode()

            def send(data: bytes) -> None:
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                send(chunk({"role": "assistant", "content": ""}, None))
            text, thought, count, finish = "", "", 0, "length"
            splitter = ThinkSplitter(prompt)

            def emit(reasoning: str, content: str) -> None:
                if stream and reasoning:
                    send(chunk({"reasoning_content": reasoning}, None))
                if stream and content:
                    send(chunk({"content": content}, None))

            pieces = state.model.generate(prompt, budget, **search_options(body))
            try:
                for piece in pieces:
                    count += 1
                    reasoning, piece = splitter.feed(piece)
                    thought += reasoning
                    text += piece
                    hit = next((s for s in stops if s in text), None)
                    if hit is not None:
                        cut = text.index(hit)
                        piece = piece[: max(0, len(piece) - (len(text) - cut))]
                        text = text[:cut]
                        finish = "stop"
                    emit(reasoning, piece)
                    if hit is not None:
                        break
                else:
                    finish = "stop" if count < budget else "length"
                    reasoning, piece = splitter.flush()
                    thought, text = thought + reasoning, text + piece
                    emit(reasoning, piece)
            except (BrokenPipeError, ConnectionResetError):
                return  # the client left; closing the generator frees the NPU
            finally:
                pieces.close()
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": count,
                "total_tokens": prompt_tokens + count,
            }
            if not stream:
                message = {"role": "assistant", "content": text}
                if thought:
                    message["reasoning_content"] = thought
                self._json(
                    200,
                    {
                        "id": cid,
                        "object": "chat.completion",
                        "created": created,
                        "model": state.model_id,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": usage,
                    },
                )
                return
            try:
                send(chunk({}, finish))
                if usage_wanted:
                    tail = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state.model_id,
                        "choices": [],
                        "usage": usage,
                    }
                    send(f"data: {json.dumps(tail)}\n\n".encode())
                send(b"data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def exit_with_parent(parent_pid: int) -> None:
    """Free the NPU if Studio dies without stopping us (Windows has no parent-death signal)."""
    if sys.platform != "win32" or parent_pid <= 0:
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE
    if not handle:
        os._exit(0)  # already gone

    def wait() -> None:
        kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(0)

    threading.Thread(target = wait, daemon = True, name = "parent-watch").start()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument("--dll-dir", required = True)
    parser.add_argument("--model-dir", required = True)
    parser.add_argument("--model-id", required = True)
    parser.add_argument("--port", type = int, required = True)
    parser.add_argument("--context-length", type = int, required = True)
    parser.add_argument("--parent-pid", type = int, default = 0)
    args = parser.parse_args(argv)
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        print(f"{API_KEY_ENV} is not set.", file = sys.stderr)
        return 2
    exit_with_parent(args.parent_pid)
    from core.inference.ryzenai_oga import Model, Oga

    model_dir = Path(args.model_dir)
    template, tokens = load_chat_template(model_dir)
    model = Model(Oga(args.dll_dir), str(model_dir))
    state = State(model, args.model_id, args.context_length, template, tokens)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state, api_key))
    server.daemon_threads = True
    print(f"RyzenAI server ready on port {args.port}", flush = True)
    try:
        server.serve_forever()
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
