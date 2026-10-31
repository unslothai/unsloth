# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A long transcript's token count gets time to finish, so compaction can run (#15800)."""

from __future__ import annotations

import httpx
import pytest

import core.inference.llama_cpp as llama_cpp
from core.inference.checkpoint import fit_checkpoint_context
from core.inference.context_window import fit_rolling_context

# The reporter's llama-server: 11.1 s to tokenize a 1.68M-token prompt.
SERVER_CHARS_PER_S = 600_000


class _Backend(llama_cpp.LlamaCppBackend):
    is_loaded = True
    base_url = "http://127.0.0.1:8080"
    _auth_headers: dict = {}


def _slow_llama_server(clients: list):
    """Renders and tokenizes at SERVER_CHARS_PER_S; times out when the client's read timeout is shorter."""

    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(
            self,
            *_args,
            timeout = None,
            **_kwargs,
        ):
            self.timeout = timeout
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(
            self,
            url,
            json = None,
            **_kwargs,
        ):
            body = json or {}
            if url.endswith("/apply-template"):
                text = "|".join(str(m.get("content") or "") for m in body.get("messages", []))
                payload = {"prompt": text}
            else:
                text = body.get("content", "")
                payload = {"tokens": [0] * (len(text) // 4)}
            if len(text) / SERVER_CHARS_PER_S > self.timeout.read:
                raise httpx.ReadTimeout("timed out")
            return _Resp(payload)

    return _Client


def _transcript(turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "you are helpful"}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"question {index} " + "word " * 400})
        messages.append({"role": "assistant", "content": f"answer {index} " + "word " * 2000})
    messages.append({"role": "user", "content": "What is 2+2?"})
    return messages


# About 7.2M characters: 12 s for the stand-in, past the old fixed 10 s.
LONG = _transcript(600)


@pytest.fixture
def backend(monkeypatch):
    clients: list = []
    monkeypatch.setattr(llama_cpp.httpx, "Client", _slow_llama_server(clients))
    instance = _Backend.__new__(_Backend)
    instance.clients = clients
    return instance


def _strict_count(backend):
    return lambda messages: backend.count_chat_tokens(messages, strict = True)


def test_the_timeout_keeps_the_floor_for_a_short_prompt():
    assert llama_cpp._count_timeout([{"role": "user", "content": "hi"}], "", None) < 10.1


def test_the_timeout_leaves_a_slow_cpu_room_to_count_a_long_transcript():
    chars = sum(len(m["content"]) for m in LONG)
    assert chars / SERVER_CHARS_PER_S > 10
    assert llama_cpp._count_timeout(LONG, "", None) > 5 * chars / SERVER_CHARS_PER_S


def test_a_long_transcript_is_counted_rather_than_timing_out(backend):
    counted = backend.count_chat_tokens(LONG, strict = True)
    assert counted > 1_000_000
    timeout = backend.clients[-1].timeout
    assert timeout.read > 10
    assert (timeout.connect, timeout.write, timeout.pool) == (10, 10, 10)


def test_rolling_compacts_a_transcript_whose_count_outlasts_ten_seconds(backend):
    fitted, truncation = fit_rolling_context(
        LONG, context_length = 16384, max_tokens = 64, count_tokens = _strict_count(backend)
    )
    assert truncation["fits"] is True
    assert truncation["dropped_messages"] > 0
    assert fitted[-1]["content"] == "What is 2+2?"


@pytest.mark.parametrize("force_reset", (False, True))
def test_checkpoint_compacts_a_transcript_whose_count_outlasts_ten_seconds(backend, force_reset):
    fitted, truncation = fit_checkpoint_context(
        LONG,
        context_length = 16384,
        max_tokens = 64,
        count_tokens = _strict_count(backend),
        can_reset = True,
        searchable = True,
        force_reset = force_reset,
    )
    assert truncation["fits"] is True
    assert truncation["checkpoint_started"] is True
    assert fitted[-1]["content"] == "What is 2+2?"


def _timing_out_client():
    class _Resp:
        status_code = 200

        def json(self):
            return {"prompt": "rendered"}

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, url, **_kwargs):
            if url.endswith("/apply-template"):
                return _Resp()
            raise httpx.ReadTimeout("timed out")

    return _Client


@pytest.mark.parametrize("strict", (True, False))
def test_a_count_timeout_is_not_reported_as_a_template_refusal(monkeypatch, strict):
    monkeypatch.setattr(llama_cpp.httpx, "Client", _timing_out_client())
    backend = _Backend.__new__(_Backend)
    messages = [{"role": "user", "content": "hello"}]
    if strict:
        with pytest.raises(RuntimeError, match = "timed out counting the prompt"):
            backend.count_chat_tokens(messages, strict = True)
    else:
        assert backend.count_chat_tokens(messages) == 0
