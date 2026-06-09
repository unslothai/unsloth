# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Anthropic ``citations_delta`` handling in the streaming proxy.

Verifies the proxy injects inline ``[N]`` markers after cited text, dedupes by
type-specific anchor (char_location, page_location, content_block_location,
search_result_location), forwards a synthetic ``document_citations`` tool_event
at message_stop, and stays inert when no citations_delta events fire. See
https://platform.claude.com/docs/en/build-with-claude/citations
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _sse(events: list[dict]) -> bytes:
    out = []
    for e in events:
        ev = e.get("type", "message")
        out.append(f"event: {ev}\ndata: {json.dumps(e)}\n\n")
    return "".join(out).encode("utf-8")


def _capture(monkeypatch, events: list[dict]) -> list[str]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _sse(events),
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    lines: list[str] = []

    async def run():
        client = _make_client()
        try:
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "what color is grass?"}],
                model = "claude-opus-4-7",
                max_tokens = 64,
            ):
                lines.append(line)
        finally:
            await client.close()

    _drive(run())
    return lines


def _message_start() -> dict:
    return {
        "type": "message_start",
        "message": {
            "id": "m1",
            "content": [],
            "model": "claude-opus-4-7",
            "role": "assistant",
            "stop_reason": None,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    }


def _content_block_start_text() -> dict:
    return {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }


def _text_delta(text: str, index: int = 0) -> dict:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


def _citations_delta(citation: dict, index: int = 0) -> dict:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "citations_delta", "citation": citation},
    }


def _content_block_stop(index: int = 0) -> dict:
    return {"type": "content_block_stop", "index": index}


def _message_delta_end() -> dict:
    return {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}


def _message_stop() -> dict:
    return {"type": "message_stop"}


def _joined(lines: list[str]) -> str:
    return "\n".join(lines)


def test_no_citations_stream_unchanged(monkeypatch):
    """Plain text passes through with no inline markers or document_citations."""
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Grass is green."),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "Grass is green." in body
    assert "document_citations" not in body
    assert "[1]" not in body


def test_single_char_location_emits_inline_marker(monkeypatch):
    cit = {
        "type": "char_location",
        "cited_text": "The grass is green.",
        "document_index": 0,
        "document_title": "Example",
        "start_char_index": 0,
        "end_char_index": 20,
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Grass is green."),
            _citations_delta(cit),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "Grass is green." in body
    assert "[1]" in body, body
    assert "document_citations" in body, body
    assert '"document_index": 0' in body, body
    assert "_key" not in body, body


def test_duplicate_citation_dedupes_to_same_number(monkeypatch):
    cit = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Example",
        "start_char_index": 0,
        "end_char_index": 20,
        "cited_text": "The grass is green.",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Grass."),
            _citations_delta(cit),
            _text_delta(" Still green."),
            _citations_delta(cit),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert body.count("[1]") == 2, body
    citation_blob = body[body.index("document_citations") :]
    assert citation_blob.count('"start_char_index"') == 1, citation_blob


def test_distinct_sources_get_distinct_numbers(monkeypatch):
    cit1 = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc A",
        "start_char_index": 0,
        "end_char_index": 5,
    }
    cit2 = {
        "type": "page_location",
        "document_index": 1,
        "document_title": "Doc B",
        "start_page_number": 3,
        "end_page_number": 4,
    }
    cit3 = {
        "type": "content_block_location",
        "document_index": 2,
        "document_title": "Doc C",
        "start_block_index": 0,
        "end_block_index": 1,
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("First"),
            _citations_delta(cit1),
            _text_delta(" Second"),
            _citations_delta(cit2),
            _text_delta(" Third"),
            _citations_delta(cit3),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body and "[2]" in body and "[3]" in body, body
    assert body.index("[1]") < body.index("[2]") < body.index("[3]")


def test_search_result_location_supported(monkeypatch):
    cit = {
        "type": "search_result_location",
        "document_index": 0,
        "document_title": "Anthropic Search Results",
        "source": "https://example.com/doc.html",
        "start_block_index": 0,
        "end_block_index": 1,
        "cited_text": "blah",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Some sourced fact."),
            _citations_delta(cit),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body
    assert "search_result_location" in body
    assert "example.com/doc.html" in body


def test_same_start_different_end_offsets_get_distinct_numbers(monkeypatch):
    """Same start_char_index + different end_char_index = distinct spans,
    so they must get distinct footnote numbers (ranges use exclusive end)."""
    cit_a = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 100,
        "end_char_index": 150,
        "cited_text": "first half",
    }
    cit_b = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 100,
        "end_char_index": 250,
        "cited_text": "wider span",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("A "),
            _citations_delta(cit_a),
            _text_delta(" and B "),
            _citations_delta(cit_b),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body, body
    assert "[2]" in body, body


def test_search_result_location_different_indices_get_distinct_numbers(monkeypatch):
    """Same source + different search_result_index = distinct footnotes
    (matches the Anthropic search-result citation contract)."""
    cit_a = {
        "type": "search_result_location",
        "search_result_index": 0,
        "source": "https://example.com/result.html",
        "title": "Result",
        "start_block_index": 0,
        "end_block_index": 1,
        "cited_text": "first",
    }
    cit_b = {
        "type": "search_result_location",
        "search_result_index": 1,
        "source": "https://example.com/result.html",
        "title": "Result",
        "start_block_index": 0,
        "end_block_index": 1,
        "cited_text": "second",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("A "),
            _citations_delta(cit_a),
            _text_delta(" and B "),
            _citations_delta(cit_b),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body, body
    assert "[2]" in body, body
