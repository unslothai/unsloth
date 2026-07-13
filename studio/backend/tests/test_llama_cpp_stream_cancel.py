# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import contextlib
import os
import sys
import threading

import httpx
import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.llama_cpp import LlamaCppBackend, _LlamaStreamCancelled


def _backend_stub() -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48848
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    return backend


def test_stream_cancel_uses_internal_exception_not_generator_exit():
    class FakeResponse:
        status_code = 200

        def close(self):
            pass

    class FakeStream:
        def __enter__(self):
            return FakeResponse()

        def __exit__(self, *_args):
            return False

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeStream()

    cancel_event = threading.Event()

    with pytest.raises(Exception) as exc_info:
        with LlamaCppBackend._stream_with_retry(
            FakeClient(),
            "http://llama.test/v1/chat/completions",
            {},
            cancel_event,
        ):
            cancel_event.set()
            raise httpx.ReadError("client closed")

    assert exc_info.type is _LlamaStreamCancelled
    assert not issubclass(exc_info.type, GeneratorExit)


def test_generate_chat_completion_swallows_internal_stream_cancel(monkeypatch):
    backend = _backend_stub()

    @contextlib.contextmanager
    def fake_open_stream(*_args, **_kwargs):
        raise _LlamaStreamCancelled

    monkeypatch.setattr(backend, "_open_stream", fake_open_stream)

    chunks = list(
        backend.generate_chat_completion(
            [{"role": "user", "content": "hi"}],
            cancel_event=threading.Event(),
        )
    )

    assert chunks == []
