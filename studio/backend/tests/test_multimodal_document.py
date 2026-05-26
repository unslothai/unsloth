# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for PDF / document attachment translation on external providers.

Studio introduces a normalised `input_document` content part on
ChatCompletionRequest so the frontend doesn't have to know the
per-provider attachment shape:

- Anthropic: translates to `{type:"document", source:{type:"base64"|"url", ...}}`
- OpenAI Responses: translates to `{type:"input_file", file_data|file_url, filename?}`

These tests pin the translation shape on both paths for base64 data
URIs and remote URLs, with optional filename metadata, and confirm
unknown / empty document parts are dropped without breaking the
request.
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _capture(monkeypatch, *, provider: str, base_url: str, messages) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        if provider == "anthropic":
            body = b"event: message_stop\n" b'data: {"type": "message_stop"}\n\n'
        else:
            body = (
                b"event: response.completed\n"
                b'data: {"type":"response.completed",'
                b'"response":{"output":[],"usage":{"input_tokens":0,'
                b'"output_tokens":0}}}\n\n'
            )
        return httpx.Response(
            200,
            content = body,
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = provider,
            base_url = base_url,
            api_key = "sk-test",
        )
        kwargs = {
            "messages": messages,
            "model": "claude-opus-4-7" if provider == "anthropic" else "gpt-5.5",
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 32,
        }
        if provider == "openai":
            kwargs["reasoning_effort"] = "medium"
        async for _ in client.stream_chat_completion(**kwargs):
            pass
        await client.close()

    _drive(run())
    return captured


_TINY_PDF_B64 = "JVBERi0xLjQKJcOkw7zDtsOfCjEgMCBvYmoKPDw+PgplbmRvYmoK"
_PDF_DATA_URI = f"data:application/pdf;base64,{_TINY_PDF_B64}"


# ── Anthropic translation ───────────────────────────────────────────


def _strip_cache(p: dict) -> dict:
    # Studio's prompt-cache wiring attaches cache_control:{type:ephemeral}
    # to the tail block of the last user message; strip it before
    # comparing the document core fields so this test stays focused
    # on the translation, not the caching layer.
    return {k: v for k, v in p.items() if k != "cache_control"}


def test_anthropic_base64_pdf_becomes_document_block(monkeypatch):
    captured = _capture(
        monkeypatch,
        provider = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarise this paper."},
                    {
                        "type": "input_document",
                        "file_data": _PDF_DATA_URI,
                        "filename": "paper.pdf",
                    },
                ],
            }
        ],
    )
    user_msg = captured["body"]["messages"][0]
    parts = user_msg["content"]
    types = [p.get("type") for p in parts]
    assert "document" in types, parts
    doc = _strip_cache(next(p for p in parts if p.get("type") == "document"))
    # citations: {enabled: true} opts into Anthropic's natural-citation
    # pipeline; without it the citations_delta handler is a no-op.
    assert doc == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": _TINY_PDF_B64,
        },
        "citations": {"enabled": True},
        "title": "paper.pdf",
    }


def test_anthropic_url_pdf_becomes_document_block(monkeypatch):
    captured = _capture(
        monkeypatch,
        provider = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this URL."},
                    {
                        "type": "input_document",
                        "file_url": "https://example.com/doc.pdf",
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    doc = _strip_cache(next(p for p in parts if p.get("type") == "document"))
    assert doc == {
        "type": "document",
        "source": {"type": "url", "url": "https://example.com/doc.pdf"},
        "citations": {"enabled": True},
    }


def test_anthropic_empty_document_part_is_dropped(monkeypatch):
    captured = _capture(
        monkeypatch,
        provider = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi."},
                    {"type": "input_document"},  # nothing usable
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    types = [p.get("type") for p in parts]
    assert "document" not in types, parts


def test_anthropic_empty_only_document_drops_whole_message(monkeypatch):
    # If the ONLY part in a user message is an unparseable input_document,
    # the helper must NOT append an empty-content message to the outbound
    # body (Anthropic 400s on "at least one block is required").
    captured = _capture(
        monkeypatch,
        provider = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        messages = [
            {"role": "user", "content": [{"type": "input_document"}]},
            {"role": "user", "content": "but THIS one is fine"},
        ],
    )
    msgs = captured["body"]["messages"]
    # The empty-content message must be skipped; only the second remains.
    assert len(msgs) == 1, msgs


def test_anthropic_empty_data_uri_payload_is_dropped(monkeypatch):
    # Codex P2: `data:application/pdf;base64,` with no payload (or
    # whitespace-only) would create an empty `source.data` that
    # Anthropic 400s on. Must be filtered before the wire.
    captured = _capture(
        monkeypatch,
        provider = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "still here"},
                    {
                        "type": "input_document",
                        "file_data": "data:application/pdf;base64,",
                        "filename": "empty.pdf",
                    },
                    {
                        "type": "input_document",
                        "file_data": "data:application/pdf;base64,   ",
                        "filename": "whitespace.pdf",
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    assert all(p.get("type") != "document" for p in parts), parts


def test_anthropic_empty_data_uri_falls_back_to_file_url(monkeypatch):
    # Codex P2 follow-up: my previous fix added the empty-data-URI ->
    # file_url fallback to the OpenAI side but missed the Anthropic
    # side, where the empty-payload branch did `continue` and discarded
    # an otherwise-valid file_url on the same part. Mirror the OpenAI
    # behavior so a malformed inline payload + remote URL still
    # attaches.
    captured = _capture(
        monkeypatch,
        provider = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this."},
                    {
                        "type": "input_document",
                        "file_data": "data:application/pdf;base64,",
                        "file_url": "https://example.com/doc.pdf",
                        "filename": "doc.pdf",
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    doc = _strip_cache(next(p for p in parts if p.get("type") == "document"))
    # base64 source MUST NOT have landed on the wire; URL source survived.
    assert doc == {
        "type": "document",
        "source": {"type": "url", "url": "https://example.com/doc.pdf"},
        "citations": {"enabled": True},
        "title": "doc.pdf",
    }


def test_anthropic_whitespace_only_data_uri_falls_back_to_file_url(monkeypatch):
    captured = _capture(
        monkeypatch,
        provider = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this."},
                    {
                        "type": "input_document",
                        "file_data": "data:application/pdf;base64,    ",
                        "file_url": "https://example.com/doc.pdf",
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["messages"][0]["content"]
    doc = _strip_cache(next(p for p in parts if p.get("type") == "document"))
    assert doc == {
        "type": "document",
        "source": {"type": "url", "url": "https://example.com/doc.pdf"},
        "citations": {"enabled": True},
    }


# ── OpenAI Responses translation ────────────────────────────────────


def test_openai_base64_pdf_becomes_input_file(monkeypatch):
    captured = _capture(
        monkeypatch,
        provider = "openai",
        base_url = "https://api.openai.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarise this paper."},
                    {
                        "type": "input_document",
                        "file_data": _PDF_DATA_URI,
                        "filename": "paper.pdf",
                    },
                ],
            }
        ],
    )
    user_msg = captured["body"]["input"][0]
    parts = user_msg["content"]
    fileblk = next(p for p in parts if p.get("type") == "input_file")
    assert fileblk == {
        "type": "input_file",
        "file_data": _PDF_DATA_URI,
        "filename": "paper.pdf",
    }


def test_openai_url_pdf_becomes_input_file(monkeypatch):
    captured = _capture(
        monkeypatch,
        provider = "openai",
        base_url = "https://api.openai.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this URL."},
                    {
                        "type": "input_document",
                        "file_url": "https://example.com/doc.pdf",
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["input"][0]["content"]
    fileblk = next(p for p in parts if p.get("type") == "input_file")
    assert fileblk == {
        "type": "input_file",
        "file_url": "https://example.com/doc.pdf",
    }


def test_openai_empty_data_uri_falls_back_to_file_url(monkeypatch):
    # Codex P2 follow-up: an empty `data:application/pdf;base64,`
    # payload was being preferred over a perfectly valid `file_url`
    # in the same part, sending `file_data=""` to OpenAI and 400ing
    # the whole turn. The translator must treat empty data URIs as
    # missing and recover via file_url.
    captured = _capture(
        monkeypatch,
        provider = "openai",
        base_url = "https://api.openai.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this."},
                    {
                        "type": "input_document",
                        "file_data": "data:application/pdf;base64,",
                        "file_url": "https://example.com/doc.pdf",
                        "filename": "doc.pdf",
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["input"][0]["content"]
    fileblk = next(p for p in parts if p.get("type") == "input_file")
    # file_data MUST NOT be on the wire; file_url survives.
    assert "file_data" not in fileblk, fileblk
    assert fileblk["file_url"] == "https://example.com/doc.pdf"
    assert fileblk["filename"] == "doc.pdf"


def test_openai_whitespace_only_data_uri_falls_back_to_file_url(monkeypatch):
    captured = _capture(
        monkeypatch,
        provider = "openai",
        base_url = "https://api.openai.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this."},
                    {
                        "type": "input_document",
                        "file_data": "data:application/pdf;base64,    ",
                        "file_url": "https://example.com/doc.pdf",
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["input"][0]["content"]
    fileblk = next(p for p in parts if p.get("type") == "input_file")
    assert "file_data" not in fileblk, fileblk
    assert fileblk["file_url"] == "https://example.com/doc.pdf"


def test_openai_empty_data_uri_without_fallback_is_dropped(monkeypatch):
    # If the only signal is an empty data URI (no file_url), the
    # whole part is skipped rather than sent as `file_data=""`.
    captured = _capture(
        monkeypatch,
        provider = "openai",
        base_url = "https://api.openai.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi."},
                    {
                        "type": "input_document",
                        "file_data": "data:application/pdf;base64,",
                        "filename": "empty.pdf",
                    },
                ],
            }
        ],
    )
    parts = captured["body"]["input"][0]["content"]
    types = [p.get("type") for p in parts]
    assert "input_file" not in types, parts


def test_openai_empty_document_part_is_dropped(monkeypatch):
    captured = _capture(
        monkeypatch,
        provider = "openai",
        base_url = "https://api.openai.com/v1",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi."},
                    {"type": "input_document"},
                ],
            }
        ],
    )
    parts = captured["body"]["input"][0]["content"]
    types = [p.get("type") for p in parts]
    assert "input_file" not in types, parts


# ── Pydantic schema + builder pass-through ──────────────────────────
#
# The translation tests above call the external-provider client directly
# with hand-built dicts, which bypasses BOTH ChatCompletionRequest's
# discriminated Union AND routes/inference._build_external_messages. The
# tests below close that gap: parse an input_document part through the
# real request schema, run the builder, and assert the part survives to
# the dict the client would receive.


def test_chat_message_accepts_input_document_part():
    from models.inference import ChatMessage

    msg = ChatMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "input_document",
                    "file_data": _PDF_DATA_URI,
                    "filename": "paper.pdf",
                    "media_type": "application/pdf",
                },
            ],
        }
    )
    assert isinstance(msg.content, list)
    assert msg.content[1].type == "input_document"
    assert msg.content[1].file_data == _PDF_DATA_URI
    assert msg.content[1].filename == "paper.pdf"
    assert msg.content[1].media_type == "application/pdf"


def test_build_external_messages_passes_input_document_for_anthropic_and_openai():
    # Both providers' stream helpers have explicit input_document
    # translation logic (Anthropic -> {type:"document"}, OpenAI
    # Responses -> {type:"input_file"}), so the part round-trips
    # through the builder unchanged on those routes.
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarise"},
                    {
                        "type": "input_document",
                        "file_url": "https://example.com/doc.pdf",
                        "filename": "doc.pdf",
                    },
                ],
            }
        )
    ]
    for provider in ("anthropic", "openai"):
        out = _build_external_messages(
            msgs, supports_vision = True, provider_type = provider
        )
        assert len(out) == 1, (provider, out)
        parts = out[0]["content"]
        assert parts[0] == {"type": "text", "text": "summarise"}, provider
        assert parts[1] == {
            "type": "input_document",
            "file_url": "https://example.com/doc.pdf",
            "filename": "doc.pdf",
        }, provider


def test_build_external_messages_strips_input_document_for_unmapped_providers():
    # Codex P1 follow-up: gemini / mistral / kimi / openrouter / deepseek
    # / custom go through generic /chat/completions passthrough that
    # forwards `messages` verbatim. Handing them an `input_document`
    # part fails the upstream validator. Builder must strip the part
    # for every provider whose stream helper doesn't translate it.
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarise"},
                    {
                        "type": "input_document",
                        "file_url": "https://example.com/doc.pdf",
                        "filename": "doc.pdf",
                    },
                ],
            }
        )
    ]
    for provider in ("gemini", "mistral", "kimi", "openrouter", "deepseek", "qwen"):
        out = _build_external_messages(
            msgs, supports_vision = True, provider_type = provider
        )
        assert len(out) == 1, (provider, out)
        parts = out[0]["content"]
        types = [p.get("type") for p in parts if isinstance(p, dict)]
        assert "input_document" not in types, (provider, parts)
        # Text part survives.
        assert {"type": "text", "text": "summarise"} in parts, (provider, parts)


def test_build_external_messages_strips_input_document_when_provider_type_unknown():
    # Defensive: legacy callers that don't pass provider_type must
    # not leak the part to an unknown destination.
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarise"},
                    {
                        "type": "input_document",
                        "file_data": _PDF_DATA_URI,
                    },
                ],
            }
        )
    ]
    out = _build_external_messages(msgs, supports_vision = True)
    parts = out[0]["content"]
    types = [p.get("type") for p in parts if isinstance(p, dict)]
    assert "input_document" not in types, parts


def test_build_external_messages_drops_input_document_for_non_vision_provider():
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    msgs = [
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarise"},
                    {
                        "type": "input_document",
                        "file_data": _PDF_DATA_URI,
                    },
                ],
            }
        )
    ]
    out = _build_external_messages(msgs, supports_vision = False)
    assert out == [{"role": "user", "content": "summarise"}]
