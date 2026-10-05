# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from starlette.requests import Request

import routes.inference as ri
from core.inference.context_window import estimate_messages_tokens_conservative
from models.inference import ChatCompletionRequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_provider_sampling_over_the_wire import _Handler, _run, _Server  # noqa: E402


_TURN = "word " * 400


def _long_chat(turns: int = 12, turn: str = _TURN) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "Answer in French."}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"question {index} {turn}"})
        messages.append({"role": "assistant", "content": f"answer {index} {turn}"})
    messages.append({"role": "user", "content": "latest question"})
    return messages


class _AnthropicCompactingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs) -> None:
        return

    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "compaction", "content": None},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": "Earlier: twelve questions."},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Bonjour"},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
        {"type": "message_stop"},
    ]

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.server.recorded.append(json.loads(self.rfile.read(length) or b"{}"))  # type: ignore[attr-defined]
        sse = "".join(
            f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in self.events
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


class _AnthropicQuotingHandler(_AnthropicCompactingHandler):
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": 'The event is called "compaction_block'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
        {"type": "message_stop"},
    ]


def _proxy(
    provider_type: str,
    messages: list[dict],
    handler = _Handler,
    **fields,
):
    async def receive() -> dict:
        return {"type": "http.disconnect"}

    chunks: list[dict] = []
    with _Server(handler = handler) as server:
        payload = ChatCompletionRequest(
            provider_type = provider_type,
            provider_base_url = server.base_url,
            messages = messages,
            model = fields.pop("model", "a-model"),
            stream = True,
            max_tokens = fields.pop("max_tokens", 256),
            **fields,
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"content-type", b"application/json")],
                "query_string": b"",
            },
            receive,
        )

        async def go() -> None:
            response = await ri._proxy_to_external_provider(payload, request)
            async for part in response.body_iterator:
                text = part.decode() if isinstance(part, bytes) else part
                for line in text.splitlines():
                    body = line[len("data:") :].strip() if line.startswith("data:") else ""
                    if body and body != "[DONE]":
                        chunks.append(json.loads(body))

        _run(go)
        assert server.bodies, "the route never reached the provider"
        return chunks, server.bodies[-1]


def _truncations(chunks: list[dict]) -> list[dict]:
    return [c["context_truncated"] for c in chunks if "context_truncated" in c]


def test_auto_compact_drops_the_oldest_turns_before_a_provider_without_compaction():
    chunks, sent = _proxy(
        "openrouter",
        _long_chat(),
        context_overflow = "truncate_oldest",
        compaction_threshold = 6_000,
    )
    roles = [m["role"] for m in sent["messages"]]
    assert len(sent["messages"]) < len(_long_chat())
    assert roles[0] == "system" and "Answer in French." in json.dumps(sent["messages"][0])
    assert sent["messages"][-1]["content"] == "latest question"
    [truncation] = _truncations(chunks)
    assert truncation["fits"] is True
    assert truncation["dropped_messages"] == len(_long_chat()) - len(sent["messages"])


def test_auto_compact_off_forwards_the_whole_chat():
    chunks, sent = _proxy("openrouter", _long_chat())
    assert len(sent["messages"]) == len(_long_chat())
    assert _truncations(chunks) == []


def test_a_chat_within_the_threshold_is_forwarded_untouched():
    short = _long_chat(turns = 1)
    chunks, sent = _proxy(
        "openrouter", short, context_overflow = "truncate_oldest", compaction_threshold = 6_000
    )
    assert len(sent["messages"]) == len(short)
    assert _truncations(chunks) == []


def test_a_provider_compaction_is_reported_as_a_summarized_compaction():
    chat = _long_chat(turns = 2)
    chunks, sent = _proxy(
        "anthropic",
        chat,
        handler = _AnthropicCompactingHandler,
        model = "claude-sonnet-4-6",
        context_overflow = "truncate_oldest",
        compaction_threshold = 150_000,
    )
    assert sent["context_management"]["edits"][0]["trigger"]["value"] == 150_000
    assert len(sent["messages"]) == len(chat) - 1
    [truncation] = _truncations(chunks)
    assert truncation == {
        "dropped_messages": 4,
        "boundary_messages": 4,
        "fits": True,
        "summarized": True,
    }


def test_model_text_naming_a_compaction_block_is_not_a_compaction():
    chunks, _ = _proxy(
        "anthropic",
        _long_chat(turns = 2),
        handler = _AnthropicQuotingHandler,
        model = "claude-sonnet-4-6",
        context_overflow = "truncate_oldest",
        compaction_threshold = 150_000,
    )
    assert _truncations(chunks) == []


def test_the_trimmed_prompt_leaves_the_whole_reply_room_in_the_window():
    chat = _long_chat(turns = 9, turn = "word " * 370)
    assert 16_384 - 8_192 < estimate_messages_tokens_conservative(chat) < 9_216
    chunks, sent = _proxy(
        "openrouter",
        chat,
        max_tokens = 8_192,
        context_overflow = "truncate_oldest",
        compaction_threshold = 12_288,
        context_window = 16_384,
    )
    assert sent["max_tokens"] == 8_192
    assert estimate_messages_tokens_conservative(sent["messages"]) + 8_192 < 16_384
    assert sent["messages"][-1]["content"] == "latest question"
    [truncation] = _truncations(chunks)
    assert truncation["dropped_messages"] == len(chat) - len(sent["messages"])


def test_a_reply_cap_near_the_window_is_lowered_rather_than_emptying_the_chat():
    chat = _long_chat(turns = 9, turn = "word " * 370)
    _, sent = _proxy(
        "openrouter",
        chat,
        max_tokens = 16_000,
        context_overflow = "truncate_oldest",
        compaction_threshold = 12_288,
        context_window = 16_384,
    )
    assert sum(m["content"].startswith("question") for m in sent["messages"]) >= 3
    assert sent["max_tokens"] <= 16_384 // 2
    assert estimate_messages_tokens_conservative(sent["messages"]) + sent["max_tokens"] < 16_384


def test_a_window_without_auto_compact_changes_nothing():
    chat = _long_chat(turns = 9, turn = "word " * 370)
    chunks, sent = _proxy("openrouter", chat, max_tokens = 16_000, context_window = 16_384)
    assert sent["max_tokens"] == 16_000
    assert len(sent["messages"]) == len(chat)
    assert _truncations(chunks) == []
