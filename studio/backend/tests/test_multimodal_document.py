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
    assert doc == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": _TINY_PDF_B64,
        },
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
