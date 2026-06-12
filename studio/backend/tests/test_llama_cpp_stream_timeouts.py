# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for llama-server streaming timeout policy."""

import httpx
import pytest
import socket
import threading
from types import SimpleNamespace

from core.inference import llama_cpp
from core.inference.llama_cpp import LlamaCppBackend


_TEST_WORKER_TIMEOUT_S = 2.0


class _StallingTextIterator:
    def __init__(self):
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls == 1:
            return "data: first-token\n"
        raise httpx.ReadTimeout("inter-token stall")


class _FakeResponse:
    def __init__(self):
        self.closed = False
        self.request = type(
            "Request",
            (),
            {"extensions": {"timeout": {"connect": 30, "read": 600.0}}},
        )()

    def iter_text(self):
        return _StallingTextIterator()

    def close(self):
        self.closed = True


class _NeverFirstTokenIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise httpx.ReadTimeout("first token stall")


class _NoFirstTokenResponse(_FakeResponse):
    def iter_text(self):
        return _NeverFirstTokenIterator()


def test_iter_text_cancellable_raises_after_inter_token_stall(monkeypatch):
    times = iter([0.0, 1.0, 122.0])
    monkeypatch.setattr(llama_cpp.time, "monotonic", lambda: next(times))

    iterator = LlamaCppBackend._iter_text_cancellable(_FakeResponse(), stall_timeout_s = 120.0)

    assert next(iterator) == "data: first-token\n"
    with pytest.raises(httpx.ReadTimeout, match = "stopped producing tokens"):
        next(iterator)


def test_iter_text_cancellable_uses_shared_first_token_deadline(monkeypatch):
    monkeypatch.setattr(llama_cpp.time, "monotonic", lambda: 701.0)

    iterator = LlamaCppBackend._iter_text_cancellable(
        _NoFirstTokenResponse(),
        first_token_deadline = 700.0,
    )

    with pytest.raises(httpx.ReadTimeout, match = "first token stall"):
        next(iterator)


def test_stream_read_timeout_is_lowered_after_headers_arrive():
    response = _FakeResponse()

    LlamaCppBackend._set_stream_read_timeout(response, 0.5)

    assert response.request.extensions["timeout"]["read"] == 0.5


class _BlockingBeforeHeadersStream:
    def __init__(self, client):
        self.client = client

    def __enter__(self):
        self.client.entered.set()
        if not self.client.closed.wait(timeout = _TEST_WORKER_TIMEOUT_S):
            raise AssertionError("request was not closed during prefill cancel")
        raise httpx.ReadError("closed before response headers")

    def __exit__(self, *exc):
        return False


class _FakeSocket:
    def __init__(self):
        self.shutdown_called = threading.Event()
        self.close_called = threading.Event()

    def shutdown(self, how):
        if how == socket.SHUT_RDWR:
            self.shutdown_called.set()

    def close(self):
        self.close_called.set()


class _BlockingBeforeHeadersClient:
    def __init__(self):
        self.entered = threading.Event()
        self.closed = threading.Event()
        self.socket = _FakeSocket()
        self._transport = SimpleNamespace(
            _pool = SimpleNamespace(
                _connections = [
                    SimpleNamespace(
                        _connection = SimpleNamespace(
                            _network_stream = SimpleNamespace(_sock = self.socket)
                        )
                    )
                ]
            )
        )

    def stream(self, *args, **kwargs):
        return _BlockingBeforeHeadersStream(self)

    def close(self):
        self.closed.set()


def test_stream_with_retry_closes_client_when_cancelled_before_headers():
    client = _BlockingBeforeHeadersClient()
    cancel_event = threading.Event()
    caught: list[BaseException] = []

    def _run():
        try:
            with LlamaCppBackend._stream_with_retry(
                client,
                "http://llama.test/v1/chat/completions",
                {"stream": True},
                cancel_event,
            ):
                pass
        except BaseException as exc:
            caught.append(exc)

    worker = threading.Thread(target = _run)
    worker.start()
    assert client.entered.wait(timeout = _TEST_WORKER_TIMEOUT_S)

    cancel_event.set()
    worker.join(timeout = _TEST_WORKER_TIMEOUT_S)

    assert not worker.is_alive()
    assert client.socket.shutdown_called.is_set()
    assert client.socket.close_called.is_set()
    assert client.closed.is_set()
    assert caught and isinstance(caught[0], GeneratorExit)
