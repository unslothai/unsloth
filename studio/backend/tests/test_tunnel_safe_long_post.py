# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load slower than a proxy's idle timer still reaches the client.

Cloudflare quick tunnels (`--secure` mode) drop a request whose origin has sent
no body bytes for ~100s, and a 600 GB GGUF load runs for 100-330s: the browser
reported "Request failed" on loads the server completed with a 200.

Measured against a real quick tunnel before choosing this design:

  * no body at all for 150s                      -> 524
  * status line + headers at t=0, no body         -> 524 (headers are NOT enough)
  * one byte at t=90s, then silence               -> connection killed ~125s later,
                                                     client sees 200 with an EMPTY body
  * one space every 20s                           -> survives, body intact

So the padding has to be continuous, and because the status is committed before
the work finishes, a failure discovered later can only travel in the body.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


@pytest.fixture
def route(monkeypatch):
    from routes import inference as route_mod

    # Keep the test fast while preserving the ordering the real values encode.
    monkeypatch.setattr(route_mod, "_TUNNEL_KEEPALIVE_AFTER_S", 0.05)
    monkeypatch.setattr(route_mod, "_TUNNEL_KEEPALIVE_EVERY_S", 0.02)
    return route_mod


async def _collect(response):
    """Chunks a client would receive, in order."""
    return [chunk async for chunk in response.body_iterator]


def test_a_fast_call_keeps_the_plain_response(route):
    async def quick():
        return {"status": "loaded"}

    result = asyncio.run(route._tunnel_safe_json(quick(), label = "t"))
    # Not a Response: FastAPI serialises it through response_model as before.
    assert result == {"status": "loaded"}


def test_a_fast_failure_keeps_its_status_code(route):
    async def boom():
        raise HTTPException(status_code = 409, detail = "busy")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(route._tunnel_safe_json(boom(), label = "t"))
    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == "busy"


def test_a_slow_call_pads_then_sends_valid_json(route):
    async def slow():
        await asyncio.sleep(0.2)
        return {"status": "loaded", "model": "unsloth/Kimi-K3-GGUF"}

    async def run():
        response = await route._tunnel_safe_json(slow(), label = "t")
        return response, await _collect(response)

    response, chunks = asyncio.run(run())
    assert response.media_type == "application/json"
    assert response.headers["x-accel-buffering"] == "no"
    # Padding first, real payload last, and it keeps arriving until then.
    assert chunks[0] == b" "
    assert len(chunks) > 2, "a call this slow must be padded more than once"
    assert all(c == b" " for c in chunks[:-1])
    body = b"".join(chunks)
    # Leading whitespace is legal JSON, which is why no client had to change.
    assert json.loads(body) == {"status": "loaded", "model": "unsloth/Kimi-K3-GGUF"}


def test_a_slow_failure_travels_in_the_body(route):
    async def slow_boom():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code = 500, detail = "llama-server died")

    async def run():
        return await _collect(await route._tunnel_safe_json(slow_boom(), label = "t"))

    body = json.loads(b"".join(asyncio.run(run())))
    assert body["_deferred_error"] == {"status_code": 500, "detail": "llama-server died"}


def test_an_unexpected_slow_failure_becomes_a_500(route):
    async def slow_boom():
        await asyncio.sleep(0.2)
        raise RuntimeError("out of VRAM")

    async def run():
        return await _collect(await route._tunnel_safe_json(slow_boom(), label = "t"))

    body = json.loads(b"".join(asyncio.run(run())))
    assert body["_deferred_error"]["status_code"] == 500
    assert "out of VRAM" in body["_deferred_error"]["detail"]


def test_the_work_survives_a_client_disconnect(route):
    """Abandoning the stream must not cancel the load: the model still lands."""
    finished = []

    async def slow():
        await asyncio.sleep(0.2)
        finished.append(True)
        return {"status": "loaded"}

    async def run():
        response = await route._tunnel_safe_json(slow(), label = "t")
        it = response.body_iterator
        assert await it.__anext__() == b" "  # one pad, then the client vanishes
        await it.aclose()
        await asyncio.sleep(0.4)

    asyncio.run(run())
    assert finished == [True]


def test_the_pad_interval_stays_inside_the_proxy_window():
    """The tunnel kills a connection ~100s after the last byte, so leave margin.

    Reads the source rather than the module so the ``route`` fixture's fast
    values cannot mask a bad default.
    """
    src = (_backend_root / "routes" / "inference.py").read_text(encoding = "utf-8")
    after = float(src.split("_TUNNEL_KEEPALIVE_AFTER_S = ")[1].split("\n")[0])
    every = float(src.split("_TUNNEL_KEEPALIVE_EVERY_S = ")[1].split("\n")[0])
    assert after + every < 90.0, "first pad must land well before the ~100s cutoff"
    assert every < 90.0, "every subsequent gap resets the timer, so it must too"
