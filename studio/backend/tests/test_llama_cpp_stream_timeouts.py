# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for llama-server streaming timeout policy."""

import httpx
import pytest

from core.inference import llama_cpp
from core.inference.llama_cpp import LlamaCppBackend


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


def test_iter_text_cancellable_raises_after_inter_token_stall(monkeypatch):
    times = iter([0.0, 1.0, 122.0])
    monkeypatch.setattr(llama_cpp.time, "monotonic", lambda: next(times))

    iterator = LlamaCppBackend._iter_text_cancellable(
        _FakeResponse(), stall_timeout_s=120.0
    )

    assert next(iterator) == "data: first-token\n"
    with pytest.raises(httpx.ReadTimeout, match="stopped producing tokens"):
        next(iterator)


def test_stream_read_timeout_is_lowered_after_headers_arrive():
    response = _FakeResponse()

    LlamaCppBackend._set_stream_read_timeout(response, 0.5)

    assert response.request.extensions["timeout"]["read"] == 0.5
