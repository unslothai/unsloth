# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two-Spark request router, against fake llama-servers on loopback.

Every test drives real sockets: fake backends from ``spark_fake_llama`` emit SSE, the
router listens on 127.0.0.1, and ``httpx`` is the client, the same client the Studio
backend uses. Health probing runs on demand (``check_health``) rather than on the
interval so nothing here depends on timing.
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional

import httpx
import pytest

from core.inference.spark_router import (
    CONVERSATION_FIELD,
    CONVERSATION_HEADER,
    SparkRouter,
    UpstreamUnreachable,
    conversation_key,
)
from .spark_fake_llama import FakeLlama, sse_contents


def run(coro):
    return asyncio.run(coro)


async def _router(
    *backends: FakeLlama,
    slots: int = 4,
    queue_limit: Optional[int] = None,
    **kw,
) -> SparkRouter:
    router = SparkRouter(health_interval = 3600.0, **kw)
    for index, fake in enumerate(backends):
        router.add_backend(
            fake.name, "127.0.0.1", fake.port, slots, primary = index == 0, queue_limit = queue_limit
        )
    await router.start()
    return router


async def _chat(
    client: httpx.AsyncClient,
    base: str,
    body: dict,
    headers: Optional[dict] = None,
) -> List[str]:
    async with client.stream(
        "POST", f"{base}/v1/chat/completions", json = body, headers = headers or {}
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        text = ""
        async for chunk in resp.aiter_bytes():
            text += chunk.decode("utf-8")
    return sse_contents(text)


def _who(frames: List[str]) -> str:
    return frames[0].rsplit("-", 1)[0]


# ── Conversation keys ─────────────────────────────────────────────────────


def test_conversation_key_precedence_and_prefix_fallback():
    body = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}]}
    assert conversation_key({CONVERSATION_HEADER: "hdr"}, dict(body, thread_id = "t")) == "hdr"
    assert (
        conversation_key({}, dict(body, **{CONVERSATION_FIELD: "thread-9"}))
        == f"{CONVERSATION_FIELD}:thread-9"
    )
    assert conversation_key({}, dict(body, session_id = "s1")) == "session_id:s1"
    prefix = conversation_key({}, body)
    assert prefix and prefix.startswith("prefix:")
    # Later turns keep the same prefix, so the same key.
    later = {
        "messages": body["messages"]
        + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}]
    }
    assert conversation_key({}, later) == prefix
    # A different first turn is a different conversation.
    other = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "bye"}]}
    assert conversation_key({}, other) != prefix
    assert conversation_key({}, {"prompt": "raw"}) is not None
    assert conversation_key({}, {"stream": True}) is None
    assert conversation_key({}, None) is None


# ── Fan-out and stickiness ───────────────────────────────────────────────


def test_keyless_requests_fan_out_across_both_backends():
    async def scenario():
        a, b = await FakeLlama("a", delay = 0.02).start(), await FakeLlama("b", delay = 0.02).start()
        router = await _router(a, b, slots = 8)
        try:
            async with httpx.AsyncClient(timeout = 10) as client:
                results = await asyncio.gather(
                    *(_chat(client, router.base_url, {"stream": True, "n": i}) for i in range(8))
                )
            served = {_who(frames) for frames in results}
            assert served == {"a", "b"}, served
            assert a.generation_count > 0 and b.generation_count > 0
            assert a.generation_count + b.generation_count == 8
            status = router.status()
            assert status["routed_keyless"] == 8 and status["routed_sticky"] == 0
            assert status["in_flight"] == 0 and status["queue_depth"] == 0
        finally:
            await router.stop()
            await a.stop()
            await b.stop()

    run(scenario())


def test_same_conversation_key_always_maps_to_the_same_backend_and_remaps_on_failure():
    async def scenario():
        a, b = await FakeLlama("a").start(), await FakeLlama("b").start()
        router = await _router(a, b)
        try:
            # Both healthy: a key is stable across many requests.
            first = router.pick("thread-1")
            assert first is not None
            for _ in range(20):
                assert router.pick("thread-1") is first
            # Many keys spread over both.
            targets = {router.pick(f"thread-{i}").name for i in range(64)}
            assert targets == {"a", "b"}
            # Through the wire, with Studio's body tag and the prefix fallback.
            async with httpx.AsyncClient(timeout = 10) as client:
                body = {
                    "messages": [{"role": "user", "content": "hi"}],
                    CONVERSATION_FIELD: "thread-1",
                }
                served = {_who(await _chat(client, router.base_url, body)) for _ in range(6)}
                assert served == {first.name}
                prefix_body = {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "q"},
                    ]
                }
                served = {_who(await _chat(client, router.base_url, prefix_body)) for _ in range(6)}
                assert len(served) == 1
            # The tag never reaches llama-server.
            for path, body, _headers in first_fake(a, b, first.name).served:
                if path.startswith("/v1/chat"):
                    assert CONVERSATION_FIELD not in body
            # One goes down: the key re-maps to the survivor, and comes back when it
            # is healthy again (consistent hashing, not a rotation).
            await router.mark_down(first, "test")
            other = router.pick("thread-1")
            assert other is not None and other is not first
            await router._record_probe(first, True, "")
            assert router.pick("thread-1") is first
        finally:
            await router.stop()
            await a.stop()
            await b.stop()

    run(scenario())


def first_fake(a: FakeLlama, b: FakeLlama, name: str) -> FakeLlama:
    return a if a.name == name else b


# ── Health-based eviction and recovery ───────────────────────────────────


def test_health_eviction_and_recovery():
    async def scenario():
        a, b = await FakeLlama("a").start(), await FakeLlama("b").start()
        router = await _router(a, b, unhealthy_after = 2)
        try:
            assert [x.name for x in router.healthy_backends()] == ["a", "b"]
            b.health_ok = False
            await router.check_health()
            assert router.get_backend("b").healthy, "one failed probe is not an eviction"
            await router.check_health()
            assert not router.get_backend("b").healthy
            async with httpx.AsyncClient(timeout = 10) as client:
                served = {
                    _who(await _chat(client, router.base_url, {"stream": True})) for _ in range(4)
                }
            assert served == {"a"}
            b.health_ok = True
            await router.check_health()
            assert router.get_backend("b").healthy
            async with httpx.AsyncClient(timeout = 10) as client:
                results = await asyncio.gather(
                    *(_chat(client, router.base_url, {"stream": True}) for _ in range(6))
                )
            assert {_who(r) for r in results} == {"a", "b"}
            # A backend whose process is gone is evicted on the failed connect.
            await b.stop()
            await router.check_health()
            await router.check_health()
            assert not router.get_backend("b").healthy
            # And put back when something answers on its port again.
            b2 = await FakeLlama("b").start(port = b.port)
            try:
                await router.check_health()
                assert router.get_backend("b").healthy
            finally:
                await b2.stop()
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


# ── Streaming pass-through ───────────────────────────────────────────────


def test_streaming_passes_chunks_through_in_order():
    async def scenario():
        a = await FakeLlama("a", chunks = 12, delay = 0.002).start()
        router = await _router(a)
        try:
            async with httpx.AsyncClient(timeout = 10) as client:
                arrivals: List[bytes] = []
                async with client.stream(
                    "POST", f"{router.base_url}/v1/chat/completions", json = {"prompt": "x"}
                ) as resp:
                    assert resp.status_code == 200
                    assert resp.headers["content-type"].startswith("text/event-stream")
                    async for chunk in resp.aiter_raw():
                        arrivals.append(chunk)
                frames = sse_contents(b"".join(arrivals).decode("utf-8"))
                assert frames == [f"a-{i}" for i in range(12)] + ["[DONE]"]
                # Relayed as it arrived, not buffered to the end.
                assert len(arrivals) > 1
                # Non-generation paths go to the primary untouched.
                props = await client.get(f"{router.base_url}/props")
                assert props.json() == {"served_by": "a"}
                assert (await client.get(f"{router.base_url}/nope")).status_code == 404
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


# ── Backpressure ─────────────────────────────────────────────────────────


def test_backpressure_caps_in_flight_at_slots_plus_queue():
    async def scenario():
        hold = asyncio.Event()
        a = await FakeLlama("a", hold = hold).start()
        router = await _router(a, slots = 1, queue_limit = 1, queue_wait_s = 5.0)
        try:
            async with httpx.AsyncClient(timeout = 10) as client:
                first = asyncio.create_task(_chat(client, router.base_url, {"stream": True}))
                await _until(lambda: router.get_backend("a").in_flight == 1)
                second = asyncio.create_task(_chat(client, router.base_url, {"stream": True}))
                await _until(lambda: router.get_backend("a").queued == 1)
                status = router.status()
                assert status["in_flight"] == 1 and status["queue_depth"] == 1
                assert (
                    status["backends"][0]["in_flight"] == 1 and status["backends"][0]["queued"] == 1
                )
                third = await client.post(
                    f"{router.base_url}/v1/chat/completions", json = {"stream": True}
                )
                assert third.status_code == 503
                assert third.headers.get("retry-after") == "1"
                assert third.json()["error"]["type"] == "spark_router"
                assert router.status()["rejected"] == 1
                hold.set()
                results = await asyncio.gather(first, second)
                assert all(r[-1] == "[DONE]" for r in results)
                assert (
                    router.get_backend("a").in_flight == 0 and router.get_backend("a").queued == 0
                )
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


# ── Failure handling ─────────────────────────────────────────────────────


def test_peer_dying_mid_stream_gives_that_client_a_clean_error_and_keeps_the_other_backend():
    async def scenario():
        a = await FakeLlama("a").start()
        b = await FakeLlama("b", chunks = 6, die_after = 2).start()
        downs: List[str] = []

        async def on_down(backend):
            downs.append(backend.name)

        router = await _router(a, b, on_backend_down = on_down)
        try:
            key_on_b = next(k for k in (f"t{i}" for i in range(200)) if router.pick(k).name == "b")
            async with httpx.AsyncClient(timeout = 10) as client:
                frames = await _chat(
                    client, router.base_url, {"prompt": "x", CONVERSATION_FIELD: key_on_b}
                )
                # The two frames that made it, then one in-band error, then a clean end.
                assert frames[:2] == ["b-0", "b-1"]
                assert len(frames) == 3 and frames[2].startswith("error:")
                assert "mid-response" in frames[2]
                assert downs == ["b"]
                assert not router.get_backend("b").healthy
                assert router.get_backend("a").healthy
                # The same conversation now lands on the survivor; the primary is untouched.
                frames = await _chat(
                    client, router.base_url, {"prompt": "x", CONVERSATION_FIELD: key_on_b}
                )
                assert frames == ["a-0", "a-1", "a-2", "a-3", "[DONE]"]
            assert router.get_backend("b").in_flight == 0
        finally:
            await router.stop()
            await a.stop()
            await b.stop()

    run(scenario())


def test_no_healthy_backend_closes_the_connection_like_a_dead_llama_server():
    async def scenario():
        a = await FakeLlama("a").start()
        router = await _router(a)
        try:
            await a.stop()
            await router.check_health()
            await router.check_health()
            assert not router.get_backend("a").healthy
            with pytest.raises(UpstreamUnreachable):
                await router.dispatch("POST", "/v1/chat/completions", {}, b"{}")
            async with httpx.AsyncClient(timeout = 10) as client:
                # No response at all, so httpx raises the same error a dead llama-server
                # produces and LlamaCppBackend._respawn_if_dead keeps working unchanged.
                with pytest.raises(httpx.RemoteProtocolError):
                    await client.post(
                        f"{router.base_url}/v1/chat/completions", json = {"prompt": "x"}
                    )
        finally:
            await router.stop()

    run(scenario())


def test_client_disconnect_mid_stream_releases_the_slot():
    async def scenario():
        a = await FakeLlama("a", chunks = 50, delay = 0.02).start()
        router = await _router(a, slots = 1)
        try:
            async with httpx.AsyncClient(timeout = 10) as client:
                async with client.stream(
                    "POST", f"{router.base_url}/v1/chat/completions", json = {"prompt": "x"}
                ) as resp:
                    async for _chunk in resp.aiter_raw():
                        break  # walk away after the first frame
            await _until(lambda: router.get_backend("a").in_flight == 0)
            await _until(lambda: a.in_flight == 0)
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


def test_status_reports_both_nodes_and_json_round_trips():
    async def scenario():
        a, b = await FakeLlama("a").start(), await FakeLlama("b").start()
        router = await _router(a, b)
        try:
            status = router.status()
            json.dumps(status)
            assert [x["name"] for x in status["backends"]] == ["a", "b"]
            assert status["backends"][0]["primary"] and not status["backends"][1]["primary"]
            assert status["healthy_backends"] == 2
            assert status["listen"] == router.base_url
        finally:
            await router.stop()
            await a.stop()
            await b.stop()

    run(scenario())
