# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A park before the first token is waited out on its keepalives, but only up to the park cap."""

import asyncio
import itertools
import time

import httpx
import pytest

import core.inference.llama_cpp as llama_mod
from core.inference.llama_cpp import LlamaCppBackend


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_text(self):
        for chunk in self._chunks:
            time.sleep(0.01)
            yield chunk


@pytest.fixture(autouse = True)
def _short_windows(monkeypatch):
    monkeypatch.setattr(llama_mod, "_SERVER_PARK_STALL_CAP_S", 0.3)
    monkeypatch.setattr(llama_mod, "_DEFAULT_FIRST_TOKEN_TIMEOUT_S", 30.0)
    monkeypatch.setattr(
        LlamaCppBackend, "_set_stream_read_timeout", staticmethod(lambda *a, **k: None)
    )


def _read(chunks):
    return LlamaCppBackend._iter_text_cancellable(
        _Stream(chunks), first_token_deadline = time.monotonic() + 30.0
    )


def test_a_park_that_never_ends_still_times_out():
    # Ten times the cap of keepalives; without the bound the stream just ends unrefused.
    forever = itertools.chain(
        [": preempted 1\n\n"], itertools.repeat(": preempt-keepalive\n\n", 300)
    )
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        for _ in _read(forever):
            pass
    assert time.monotonic() - started < 5, "every keepalive pushed the deadline out again"


def test_a_park_that_resumes_inside_the_cap_still_answers():
    data = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    chunks = [": preempted 1\n\n"] + [": preempt-keepalive\n\n"] * 5 + [": resumed 1\n\n", data]
    assert "".join(_read(chunks)).endswith(data)


class TestARawRelayParkedAfterItsFirstToken:
    """The async relay sees each keepalive as an item; only the park cap may renew on them."""

    @staticmethod
    async def _relay(items, monkeypatch, cap):
        import routes.inference as inference

        monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", cap)

        async def upstream():
            for item in items:
                await asyncio.sleep(0.01)
                yield item

        seen = []
        async for item in inference._aiter_llama_stream_items(
            upstream(),
            first_token_deadline = time.monotonic() + 30.0,
            post_first_item_read_timeout_s = 0.1,
        ):
            seen.append(item)
        return seen

    def test_a_wedged_park_times_out(self, monkeypatch):
        items = ['data: {"choices":[{"delta":{"content":"a"}}]}', ": preempted"]
        items += [": preempt-keepalive"] * 300 + ["never sent"]

        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(items, monkeypatch, cap = 0.3))

    def test_a_park_inside_the_cap_still_answers(self, monkeypatch):
        items = ['data: {"choices":[{"delta":{"content":"a"}}]}', ": preempted"]
        items += [": preempt-keepalive"] * 5 + [
            ": resumed",
            'data: {"choices":[{"delta":{"content":"b"}}]}',
        ]
        seen = asyncio.run(self._relay(items, monkeypatch, cap = 5.0))
        assert seen[-1].endswith('"b"}}]}')
