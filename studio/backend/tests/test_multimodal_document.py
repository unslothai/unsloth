# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for PDF / document attachment translation on external providers.

Unsloth adds a normalised `input_document` content part on
ChatCompletionRequest so the frontend needn't know the per-provider
attachment shape:

- Anthropic: `{type:"document", source:{type:"base64"|"url", ...}}`
- OpenAI Responses: `{type:"input_file", file_data|file_url, filename?}`

Pins the translation shape on both paths for base64 data URIs and remote
URLs (with optional filename), and confirms unknown / empty document
parts are dropped without breaking the request.
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
    # Strip the prompt-cache cache_control off the last user block so this
    # test focuses on translation, not the caching layer.
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
    # citations:{enabled:true} opts into Anthropic's citation pipeline;
    # without it the citations_delta handler is a no-op.
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
    # If the only part is an unparseable input_document, the helper must not
    # append an empty-content message (Anthropic 400s on "at least one block").
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
    # Empty-content message skipped; only the second remains.
    assert len(msgs) == 1, msgs


def test_anthropic_empty_data_uri_payload_is_dropped(monkeypatch):
    # A `data:application/pdf;base64,` with empty/whitespace payload makes an
    # empty `source.data` that Anthropic 400s on; filter it before the wire.
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
    # The empty-data-URI -> file_url fallback existed on OpenAI but not
    # Anthropic, which discarded a valid file_url on the same part. Mirror
    # OpenAI so a malformed inline payload + remote URL still attaches.
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
    # base64 source MUST NOT reach the wire; URL source survives.
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
    assert fileblk == {"type": "input_file", "file_url": "https://example.com/doc.pdf"}


def test_openai_empty_data_uri_falls_back_to_file_url(monkeypatch):
    # An empty `data:application/pdf;base64,` payload was preferred over a valid
    # `file_url` in the same part, sending `file_data=""` and 400ing. The
    # translator must treat empty data URIs as missing and recover via file_url.
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
    # file_data MUST NOT reach the wire; file_url survives.
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
    # Only signal is an empty data URI (no file_url): skip the whole part
    # rather than send `file_data=""`.
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
# The tests above call the client with hand-built dicts, bypassing the schema
# and _build_external_messages. The tests below parse an input_document part
# through the real schema + builder and assert it survives to the client dict.


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
    # Both providers' stream helpers translate input_document (Anthropic ->
    # {type:"document"}, OpenAI Responses -> {type:"input_file"}), so the
    # part round-trips through the builder unchanged on those routes.
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
    # / custom use generic /chat/completions passthrough that forwards
    # `messages` verbatim, so an `input_document` part fails the upstream
    # validator. The builder must strip it for any provider whose stream
    # helper doesn't translate it.
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
    # Defensive: legacy callers without provider_type must not leak the
    # part to an unknown destination.
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
