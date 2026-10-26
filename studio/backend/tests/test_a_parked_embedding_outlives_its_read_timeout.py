# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An /v1/embeddings prefill the server parks longer than the request's read timeout is waited
out, as a non-streaming completion is. Its two POSTs went straight to the client with a fixed
timeout, so a healthy parked request failed with ReadTimeout without asking the park probe."""

import asyncio
import inspect
import time

import httpx
import pytest

import routes.inference as inference


def _slow_client(delay_s):
    async def handler(request):
        await asyncio.sleep(delay_s)
        return httpx.Response(200, json = {"data": [{"embedding": [0.0]}]})

    return httpx.AsyncClient(transport = httpx.MockTransport(handler))


def _post(monkeypatch, parked, timeout_s = 0.1, delay_s = 0.3):
    probes = []

    def grace():
        probes.append(time.monotonic())
        return parked

    monkeypatch.setattr(inference, "_raw_park_grace", lambda *a, **k: grace)

    async def run():
        async with _slow_client(delay_s) as client:
            return await inference._nonstreaming_post_through_park(
                client,
                "http://127.0.0.1:9/v1/embeddings",
                llama_backend = object(),
                json = {"input": "x"},
                upstream_headers = {},
                timeout = httpx.Timeout(timeout_s),
            )

    return asyncio.run(run()), probes


def test_a_parked_embedding_is_waited_out_past_its_timeout(monkeypatch):
    response, probes = _post(monkeypatch, parked = True)
    assert response.status_code == 200
    assert probes, "the park probe was asked at the deadline"


def test_an_unparked_embedding_still_ends_at_its_timeout(monkeypatch):
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        _post(monkeypatch, parked = False)
    assert time.monotonic() - started < 0.25


def test_both_embedding_posts_go_through_the_park_wait():
    source = "".join(inspect.getsource(inference.openai_embeddings).split())
    assert source.count("await_nonstreaming_post_through_park(") == 2
    assert "await_client.post(" not in source
    assert source.count("timeout=httpx.Timeout(_DEFAULT_FIRST_TOKEN_TIMEOUT_S)") == 2
