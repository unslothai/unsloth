# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge-case tests for Anthropic ``citations_delta`` handling.

Complements ``test_anthropic_citations.py``. Covers malformed payloads,
unusual orderings, mixed citation types, and the ``citations:
{enabled: true}`` opt-in attached to translated ``input_document``
blocks. See
https://platform.claude.com/docs/en/build-with-claude/citations and
https://platform.claude.com/docs/en/build-with-claude/search-results.
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


# ── shared SSE harness ───────────────────────────────────────


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


def _capture(
    monkeypatch,
    events: list[dict],
    *,
    messages: list[dict] | None = None,
    captured_body: dict | None = None,
) -> list[str]:
    """Drive ``stream_chat_completion`` against a mocked Anthropic response
    and return the SSE lines. Pass ``captured_body`` to also capture the
    outgoing request body for assertions on the translated Anthropic shape.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if captured_body is not None:
            try:
                captured_body.update(json.loads(request.content.decode("utf-8")))
            except Exception:  # pragma: no cover -- diagnostic only
                pass
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
                messages = messages or [{"role": "user", "content": "what color is grass?"}],
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


def _citation_payload(body: str) -> dict:
    """Return the ``document_citations`` synthetic tool_event payload from
    the SSE body. Raises if absent."""
    assert "document_citations" in body, body
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[len("data: ") :])
        except json.JSONDecodeError:
            continue
        tool_event = payload.get("_toolEvent") if isinstance(payload, dict) else None
        if isinstance(tool_event, dict) and tool_event.get("type") == "document_citations":
            return tool_event
    raise AssertionError("document_citations event not parsed out of SSE body")


# ── edge cases ───────────────────────────────────────────────


def test_citation_with_no_preceding_text_still_emits_marker(monkeypatch):
    """citations_delta before any text_delta must not crash; marker lands
    at the start of the block."""
    cit = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "X",
        "start_char_index": 0,
        "end_char_index": 5,
        "cited_text": "x",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _citations_delta(cit),
            _text_delta("hello"),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body, body
    assert "document_citations" in body, body


def test_citations_delta_with_non_dict_citation_is_ignored(monkeypatch):
    """Non-dict ``delta.citation`` must not crash, emit a marker, or poison
    the document_citations list."""
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Hello."),
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "citations_delta", "citation": "not-a-dict"},
            },
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "Hello." in body
    assert "[1]" not in body
    assert "document_citations" not in body


def test_citations_delta_with_missing_citation_field_is_ignored(monkeypatch):
    """Missing ``citation`` field is treated like a non-dict citation: skip
    without crashing."""
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Hello."),
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "citations_delta"},
            },
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "Hello." in body
    assert "[1]" not in body
    assert "document_citations" not in body


def test_char_location_with_reversed_indices_does_not_crash(monkeypatch):
    """Malformed char_location with reversed indices must not crash; the
    dedup key accepts any int pair and still surfaces a footnote."""
    cit = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 300,
        "end_char_index": 50,
        "cited_text": "?",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Weird."),
            _citations_delta(cit),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body, body
    payload = _citation_payload(body)
    assert payload["citations"][0]["start_char_index"] == 300
    assert payload["citations"][0]["end_char_index"] == 50


def test_page_location_missing_document_index_does_not_crash(monkeypatch):
    """page_location missing ``document_index`` still produces a footnote;
    dedup key falls back to ``None`` for the missing field."""
    cit = {
        "type": "page_location",
        "document_title": "Untitled PDF",
        "start_page_number": 1,
        "end_page_number": 2,
        "cited_text": "p1",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("From the PDF:"),
            _citations_delta(cit),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body, body
    payload = _citation_payload(body)
    assert payload["citations"][0].get("document_index") is None


def test_content_block_location_with_non_int_block_index_does_not_crash(monkeypatch):
    """content_block_location with string block indices must not crash; dedup
    key tolerates non-int values."""
    cit = {
        "type": "content_block_location",
        "document_index": 0,
        "document_title": "Custom",
        "start_block_index": "0",
        "end_block_index": "1",
        "cited_text": "anything",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Cite."),
            _citations_delta(cit),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body, body
    payload = _citation_payload(body)
    assert payload["citations"][0]["start_block_index"] == "0"


def test_unknown_citation_type_falls_back_to_stringified_key(monkeypatch):
    """Unknown citation ``type`` (forward-compat) still dedupes: identical
    ones collapse, differing ones get distinct numbers."""
    cit_a = {
        "type": "future_shape_location",
        "anchor": "abc",
        "cited_text": "blah",
    }
    cit_b = {
        "type": "future_shape_location",
        "anchor": "xyz",
        "cited_text": "blah",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("A"),
            _citations_delta(cit_a),
            _text_delta(" again"),
            _citations_delta(cit_a),
            _text_delta(" B"),
            _citations_delta(cit_b),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    # cit_a dedupes onto [1], cit_b gets [2].
    assert body.count("[1]") == 2, body
    assert body.count("[2]") == 1, body
    payload = _citation_payload(body)
    assert len(payload["citations"]) == 2


def test_mixed_citation_types_same_document_get_distinct_keys(monkeypatch):
    """char_location and page_location on the same document_index are distinct
    shapes; dedup key uses citation type as its first slot."""
    cit_char = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 0,
        "end_char_index": 10,
    }
    cit_page = {
        "type": "page_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_page_number": 1,
        "end_page_number": 2,
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("char-cite"),
            _citations_delta(cit_char),
            _text_delta(" page-cite"),
            _citations_delta(cit_page),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body and "[2]" in body, body
    payload = _citation_payload(body)
    assert len(payload["citations"]) == 2


def test_cited_text_is_preserved_in_synthetic_event(monkeypatch):
    """``cited_text`` must survive into the synthetic event so the Sources
    panel can render it as a tooltip. Anthropic does not bill cited_text
    against output tokens, so preserving it is free."""
    cit = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Trustworthy Doc",
        "start_char_index": 0,
        "end_char_index": 20,
        "cited_text": "The grass is green.",
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
    payload = _citation_payload(body)
    assert payload["citations"][0]["cited_text"] == "The grass is green."


def test_internal_key_field_never_leaks_to_client(monkeypatch):
    """The internal ``_key`` dedup sentinel must be stripped before the
    synthetic event is forwarded; it is not an Anthropic field."""
    cit = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 0,
        "end_char_index": 5,
        "cited_text": "..",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("hi"),
            _citations_delta(cit),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    payload = _citation_payload(body)
    assert payload["citations"], payload
    for c in payload["citations"]:
        assert "_key" not in c, c


def test_citation_across_multiple_content_blocks_numbers_continue(monkeypatch):
    """Footnote numbering is per-message, not per-content-block: citations
    across separate blocks emit [1] then [2]."""
    cit_a = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 0,
        "end_char_index": 5,
    }
    cit_b = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 100,
        "end_char_index": 105,
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("first"),
            _citations_delta(cit_a, index = 0),
            _content_block_stop(0),
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            _text_delta(" second", index = 1),
            _citations_delta(cit_b, index = 1),
            _content_block_stop(1),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "[1]" in body and "[2]" in body, body
    assert body.index("[1]") < body.index("[2]")
    payload = _citation_payload(body)
    assert len(payload["citations"]) == 2


def test_inline_marker_lands_after_text_run(monkeypatch):
    """Inline ``[N]`` must land AFTER the cited text run: Anthropic streams
    text then citation, so the proxy emits ``"...green.[1]"`` not
    ``"[1]green"``."""
    cit = {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Doc",
        "start_char_index": 0,
        "end_char_index": 20,
        "cited_text": "grass",
    }
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Grass is green."),
            _citations_delta(cit),
            _text_delta(" Sky is blue."),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    grass = body.index("Grass is green.")
    marker = body.index("[1]")
    sky = body.index("Sky is blue.")
    assert grass < marker < sky, body


def test_no_synthetic_event_when_only_text_deltas(monkeypatch):
    """No citations_delta means no synthetic ``document_citations`` event;
    Sources panel relies on absence to suppress the section."""
    lines = _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("Just some prose. "),
            _text_delta("More prose."),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
    )
    body = _joined(lines)
    assert "document_citations" not in body
    assert "[1]" not in body


def test_input_document_translation_enables_citations(monkeypatch):
    """``input_document`` must translate to an Anthropic ``document`` block
    carrying ``citations: {enabled: true}`` (both base64 and url source
    branches) so upstream emits citations_delta."""
    captured_b64: dict = {}
    _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("ok"),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_document",
                        "file_data": "data:application/pdf;base64,QUJD",
                        "filename": "spec.pdf",
                    },
                    {"type": "text", "text": "summarise"},
                ],
            }
        ],
        captured_body = captured_b64,
    )
    user_msg = captured_b64["messages"][0]
    doc_block = next(p for p in user_msg["content"] if p.get("type") == "document")
    assert doc_block["source"]["type"] == "base64", doc_block
    assert doc_block.get("citations") == {"enabled": True}, doc_block

    captured_url: dict = {}
    _capture(
        monkeypatch,
        [
            _message_start(),
            _content_block_start_text(),
            _text_delta("ok"),
            _content_block_stop(),
            _message_delta_end(),
            _message_stop(),
        ],
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_document",
                        "file_url": "https://example.com/doc.pdf",
                        "filename": "doc.pdf",
                    },
                    {"type": "text", "text": "summarise"},
                ],
            }
        ],
        captured_body = captured_url,
    )
    user_msg = captured_url["messages"][0]
    doc_block = next(p for p in user_msg["content"] if p.get("type") == "document")
    assert doc_block["source"]["type"] == "url", doc_block
    assert doc_block.get("citations") == {"enabled": True}, doc_block


# ── cited_text truncation + safe-url citation conversion ────────


def test_cited_text_truncated_in_synthetic_event(monkeypatch):
    """``cited_text`` is capped server-side so multi-KB spans don't balloon
    the SSE payload."""
    from core.inference.external_provider import _CITED_TEXT_MAX_LEN

    long_quote = "x" * (_CITED_TEXT_MAX_LEN + 4000)
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "claim "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "document_index": 0,
                    "document_title": "doc",
                    "start_char_index": 0,
                    "end_char_index": 5,
                    "cited_text": long_quote,
                },
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    chunks = _capture(monkeypatch, events)
    tool_events = [c for c in chunks if "_toolEvent" in c and "document_citations" in c]
    assert tool_events, "no document_citations tool event"
    payload = json.loads(tool_events[0].split("data: ", 1)[1])
    cited = payload["_toolEvent"]["citations"][0]["cited_text"]
    assert len(cited) <= _CITED_TEXT_MAX_LEN + 1, len(cited)
    assert cited.endswith("…")
