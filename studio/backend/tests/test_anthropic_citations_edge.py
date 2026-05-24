# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge-case tests for Anthropic ``citations_delta`` handling.

Complements ``test_anthropic_citations.py``. Covers malformed payloads,
unusual orderings, mixed citation types, and the ``citations: {enabled:
true}`` opt-in we now attach to translated ``input_document`` blocks.

References:
  * https://platform.claude.com/docs/en/build-with-claude/citations
  * https://platform.claude.com/docs/en/build-with-claude/search-results
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
    """Drive ``stream_chat_completion`` against a mocked Anthropic
    response and return the SSE lines we'd send back to the browser.

    Pass ``captured_body`` (a dict) to also capture the request body
    we sent upstream -- the test can then assert on the translated
    Anthropic shape (e.g. that ``citations: {enabled: true}`` made it
    onto the document block).
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
                messages = messages or [
                    {"role": "user", "content": "what color is grass?"}
                ],
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
    """Pull the ``document_citations`` synthetic tool_event out of the
    streamed SSE body and return its decoded payload.

    Raises if no such event was emitted -- helper is for the tests that
    expect the event to be present.
    """
    assert "document_citations" in body, body
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[len("data: "):])
        except json.JSONDecodeError:
            continue
        tool_event = payload.get("_toolEvent") if isinstance(payload, dict) else None
        if isinstance(tool_event, dict) and tool_event.get("type") == "document_citations":
            return tool_event
    raise AssertionError("document_citations event not parsed out of SSE body")


# ── edge cases ───────────────────────────────────────────────


def test_citation_with_no_preceding_text_still_emits_marker(monkeypatch):
    """A citations_delta arriving before any text_delta on a fresh
    content block must not crash -- the marker just lands at the
    beginning of the block. (Anthropic in practice emits text first,
    but the proxy must not assume it.)"""
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
    """A malformed event where ``delta.citation`` is not a dict (e.g.
    a future-proof string sentinel) must not crash, must not emit a
    marker, and must not poison the document_citations list."""
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
    """citations_delta with no ``citation`` field is treated the same as
    a non-dict citation -- skip without crashing."""
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
    """A malformed char_location where ``start_char_index >
    end_char_index`` must not crash the stream -- the dedup key
    accepts any int pair, and we still surface a footnote for the
    caller to inspect."""
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
    """page_location with no ``document_index`` (e.g. when Anthropic
    only sends ``document_title``) must still produce a footnote.
    The dedup key falls back to ``None`` for the missing field."""
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
    """content_block_location whose block indices are unexpectedly
    strings (future shape / provider bug) must not crash. The dedup
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
    """An unrecognised citation ``type`` (forward-compat) must still
    dedupe: identical unknown citations collapse to one footnote;
    citations that differ in any field get distinct numbers."""
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
    """A char_location and a page_location for the same
    ``document_index`` are different citation shapes and must be
    treated as distinct footnotes -- the dedup key carries the
    citation type as its first slot."""
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
    """The ``cited_text`` field on a citation must survive into the
    synthetic ``document_citations`` tool_event so the Sources panel
    can render the cited snippet as a tooltip / description.

    Per the Anthropic docs, ``cited_text`` is the verbatim cited
    span and is not counted towards output tokens, so dropping it
    here would lose user-visible context for free."""
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
    """The ``_key`` sentinel we attach to each citation for in-process
    dedup must be stripped before the synthetic event is forwarded
    -- it's internal accounting state, not a documented Anthropic
    field, and surfacing it would confuse downstream consumers."""
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
    """Footnote numbering must be per-message, not per-content-block.
    A citation on block index 0 followed by one on block index 2
    (split across a tool-use block, say) must come out as [1] then
    [2], not [1] then [1]."""
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
    """The inline ``[N]`` marker must come AFTER the text run it
    annotates, not before -- Anthropic streams the text first and
    the citation second, and the proxy preserves that order so the
    footnote reads ``"...grass is green.[1]"``, not ``"[1]...grass``."""
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
    """A stream that carries text_delta events but no citations_delta
    must NOT emit a synthetic ``document_citations`` event -- the
    Sources panel relies on absence of the event to know there are
    no document footnotes to render."""
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
    """Studio's normalised ``input_document`` content part must
    translate to an Anthropic ``document`` block carrying ``citations:
    {enabled: true}`` so the upstream actually emits citations_delta
    events. Without this opt-in the rest of the citation plumbing is
    a no-op for real user requests.

    Covers both the base64 and the url ``source`` branches."""
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
    doc_block = next(
        p for p in user_msg["content"] if p.get("type") == "document"
    )
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
    doc_block = next(
        p for p in user_msg["content"] if p.get("type") == "document"
    )
    assert doc_block["source"]["type"] == "url", doc_block
    assert doc_block.get("citations") == {"enabled": True}, doc_block
