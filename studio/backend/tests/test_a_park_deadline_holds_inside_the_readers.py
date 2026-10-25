# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The park cap runs from the park's start inside the network readers too.

A parked stream that sends keepalives almost to the cap and then falls silent used to get a fresh
cap at its next read deadline: the readers measured their grace from the moment that read timed
out, so a wedged park held its answer and its admission slot for about twice the cap. Each reader
now stops at the earlier of its own grace and the park's start plus the cap."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import httpcore
import httpx
import pytest

import core.inference.llama_cpp as llama_mod
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import ServerParkNotices
import routes.inference as inference

_CAP = 0.6
_WINDOW = 0.2
_KEEPALIVES_UNTIL = 0.5


def _script(t0):
    """A park notice at t0, keepalives to just short of the cap, then silence for good."""
    items = [(t0, b": preempted\n\n")]
    t = 0.1
    while t <= _KEEPALIVES_UNTIL + 1e-9:
        items.append((t0 + t, b": preempt-keepalive\n\n"))
        t += 0.1
    return items


class _SyncStream:
    def __init__(self, items):
        self._items = list(items)

    def read(self, max_bytes, timeout = None):
        end = time.monotonic() + (timeout if timeout is not None else 10.0)
        while True:
            now = time.monotonic()
            if self._items and self._items[0][0] <= now:
                return self._items.pop(0)[1]
            if now >= end:
                raise httpcore.ReadTimeout("silent")
            due = self._items[0][0] if self._items else end
            time.sleep(max(0.0, min(end, due) - now))


class _AsyncStream:
    def __init__(self, items):
        self._items = list(items)

    async def read(self, max_bytes, timeout = None):
        end = time.monotonic() + (timeout if timeout is not None else 10.0)
        while True:
            now = time.monotonic()
            if self._items and self._items[0][0] <= now:
                return self._items.pop(0)[1]
            if now >= end:
                raise httpcore.ReadTimeout("silent")
            due = self._items[0][0] if self._items else end
            await asyncio.sleep(max(0.0, min(end, due) - now))


def test_the_sync_reader_stops_at_the_park_cap(monkeypatch):
    monkeypatch.setattr(llama_mod, "_SERVER_PARK_STALL_CAP_S", _CAP)
    t0 = time.monotonic()
    stream = _SyncStream(_script(t0))
    client = SimpleNamespace(
        _transport = SimpleNamespace(
            _pool = SimpleNamespace(
                _connections = [
                    SimpleNamespace(_connection = SimpleNamespace(_network_stream = stream))
                ]
            )
        )
    )
    LlamaCppBackend._install_cancel_aware_read(
        client, threading.Event(), poll_s = 0.05, stall_grace = lambda: False
    )
    with pytest.raises(httpcore.ReadTimeout):
        while True:
            stream.read(65536, timeout = _WINDOW)
    waited = time.monotonic() - t0
    assert waited < _CAP + _WINDOW + 0.25, (
        f"parked {waited:.2f}s against a {_CAP}s cap: the reader restarted it at its deadline"
    )


def test_the_async_reader_stops_at_the_park_cap(monkeypatch):
    monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", _CAP)

    async def run():
        t0 = time.monotonic()
        stream = _AsyncStream(_script(t0))
        response = SimpleNamespace(
            extensions = {"network_stream": stream},
            request = SimpleNamespace(extensions = {"timeout": {"read": _WINDOW}}),
        )
        assert inference._install_park_aware_read(response, lambda: False)
        with pytest.raises(httpcore.ReadTimeout):
            while True:
                await stream.read(65536, timeout = _WINDOW)
        return time.monotonic() - t0

    waited = asyncio.run(run())
    assert waited < _CAP + _WINDOW + 0.25, (
        f"parked {waited:.2f}s against a {_CAP}s cap: the reader restarted it at its deadline"
    )


def test_the_relay_fallback_stops_at_the_park_cap(monkeypatch):
    """No wrapped read: the relay's own first-token deadline is excused from the park's start."""
    monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", _CAP)
    monkeypatch.setattr(inference, "_DEFAULT_FIRST_TOKEN_TIMEOUT_S", _WINDOW)
    monkeypatch.setattr(inference, "configured_first_token_timeout_s", lambda: _WINDOW)

    async def run():
        t0 = time.monotonic()
        items = [(t0, ": preempted")] + [
            (t0 + 0.1 * (i + 1), ": preempt-keepalive") for i in range(5)
        ]

        async def upstream():
            for due, item in items:
                await asyncio.sleep(max(0.0, due - time.monotonic()))
                yield item
            await asyncio.sleep(30)

        with pytest.raises(httpx.ReadTimeout):
            async for _ in inference._aiter_llama_stream_items(
                upstream(),
                first_token_deadline = t0 + _WINDOW,
                stall_grace = lambda: False,
            ):
                pass
        return time.monotonic() - t0

    waited = asyncio.run(run())
    assert waited < _CAP + _WINDOW + 0.25, (
        f"parked {waited:.2f}s against a {_CAP}s cap: the relay restarted it at its deadline"
    )


def test_the_park_start_is_tracked_per_park():
    notices = ServerParkNotices()
    assert notices.parked_since is None
    notices.feed(b": preempted 0\n\n")
    first = notices.parked_since
    assert first is not None
    notices.feed(b": preempted 1\n\n")
    assert notices.parked_since == first, "a second parked slot restarted the park"
    notices.feed(b": resumed 0\n\n: resumed 1\n\n")
    assert notices.parked_since is None
    time.sleep(0.01)
    notices.feed(b": preempted 0\n\n")
    assert notices.parked_since > first
