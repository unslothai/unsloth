# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two-Spark request router, against fake llama-servers on loopback.

Every test drives real sockets, with ``httpx`` as the client the Studio backend uses.
Health probing runs on demand rather than on the interval, so nothing here depends on
timing."""

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
from core.inference import spark_router as sr
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
    later = {
        "messages": body["messages"]
        + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}]
    }
    assert conversation_key({}, later) == prefix
    other = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "bye"}]}
    assert conversation_key({}, other) != prefix
    assert conversation_key({}, {"prompt": "raw"}) is not None
    assert conversation_key({}, {"stream": True}) is None
    assert conversation_key({}, None) is None


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
            first = router.pick("thread-1")
            assert first is not None
            for _ in range(20):
                assert router.pick("thread-1") is first
            targets = {router.pick(f"thread-{i}").name for i in range(64)}
            assert targets == {"a", "b"}
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
            # Consistent hashing, not a rotation: the key comes back to its own
            # backend once that backend is healthy again.
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
            await b.stop()
            await router.check_health()
            await router.check_health()
            assert not router.get_backend("b").healthy
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
                props = await client.get(f"{router.base_url}/props")
                assert props.json() == {"served_by": "a"}
                assert (await client.get(f"{router.base_url}/nope")).status_code == 404
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


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
                # httpx must raise the same error a dead llama-server produces, or
                # LlamaCppBackend._respawn_if_dead stops working.
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


def test_a_freed_slot_goes_to_the_queued_request_not_a_newcomer():
    async def scenario():
        a = await FakeLlama("a").start()
        router = await _router(a, slots = 1, queue_limit = 2)
        try:
            backend = router.get_backend("a")
            await router._acquire(backend)
            order: List[str] = []

            async def take(name: str):
                await router._acquire(backend)
                order.append(name)

            queued = asyncio.create_task(take("queued"))
            await _until(lambda: backend.queued == 1)
            await router._release(backend)
            # A newcomer arriving right as the slot frees must not jump the queue.
            newcomer = asyncio.create_task(take("newcomer"))
            await _until(lambda: len(order) == 1)
            assert order == ["queued"]
            assert backend.queued == 1 and backend.in_flight == 1
            await router._release(backend)
            await asyncio.gather(queued, newcomer)
            assert order == ["queued", "newcomer"]
            await router._release(backend)
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


def test_openai_end_user_id_is_not_a_conversation_key():
    # "user" is OpenAI's stable per-end-user abuse identifier, superseded by safety_identifier,
    # not a thread id. Keying on it collapses every conversation a person has onto one backend,
    # which on a single-user Studio session pins all traffic to one node and defeats replicas.
    a = {"user": "u-1", "messages": [{"role": "user", "content": "first topic"}]}
    b = {"user": "u-1", "messages": [{"role": "user", "content": "unrelated topic"}]}
    ka = sr.conversation_key({}, a)
    kb = sr.conversation_key({}, b)
    assert ka != kb, "two conversations from one end user collapsed onto one routing key"

    # A real conversation id still pins, and still wins over the prompt hash.
    c = {"conversation_id": "c-9", "messages": [{"role": "user", "content": "x"}]}
    d = {"conversation_id": "c-9", "messages": [{"role": "user", "content": "y"}]}
    assert sr.conversation_key({}, c) == sr.conversation_key({}, d) == "conversation_id:c-9"

    # previous_response_id is what the Responses API uses for continuity.
    e = {"previous_response_id": "resp-1", "messages": [{"role": "user", "content": "z"}]}
    assert sr.conversation_key({}, e) == "previous_response_id:resp-1"


def test_prefix_key_does_not_join_the_whole_embedding_batch():
    # Byte-identical to the eager join, without copying the batch to keep a kilobyte.
    batch = [f"row-{i}-" + "x" * 200 for i in range(5000)]
    eager = " ".join(batch)[: sr.PREFIX_KEY_CHARS]
    assert sr._join_to_limit(batch) == eager
    assert sr._prompt_prefix({"input": batch}) == eager
    # and the single-element and empty cases
    assert sr._join_to_limit(["a", "b"]) == "a b"
    assert sr._join_to_limit([]) == ""


def test_a_json_response_that_dies_mid_body_fails_the_transfer_instead_of_looking_complete():
    # /v1/embeddings, /completion and chat with stream false all answer in JSON. An SSE error
    # frame appended to one of those is not a message the client can read, and ending the
    # chunked body cleanly after it presents truncated JSON as a complete 200.
    async def scenario():
        b = await FakeLlama("b", chunks = 6, die_after = 2, content_type = "application/json").start()
        router = await _router(b)
        try:
            async with httpx.AsyncClient(timeout = 10) as client:
                with pytest.raises(httpx.HTTPError):
                    response = await client.post(
                        f"{router.base_url}/v1/chat/completions", json = {"prompt": "x"}
                    )
                    response.read()
        finally:
            await router.stop()
            await b.stop()

    run(scenario())


def test_an_sse_response_that_dies_mid_stream_still_gets_the_in_band_error():
    # The other half of the same choice: a reader of text/event-stream can be told in-band,
    # and llama-server reports its own mid-stream errors the same way.
    async def scenario():
        b = await FakeLlama("b", chunks = 6, die_after = 2).start()
        router = await _router(b)
        try:
            async with httpx.AsyncClient(timeout = 10) as client:
                frames = await _chat(client, router.base_url, {"prompt": "x"})
                assert frames[:2] == ["b-0", "b-1"]
                assert len(frames) == 3 and frames[2].startswith("error:")
        finally:
            await router.stop()
            await b.stop()

    run(scenario())


def test_an_oversized_chunked_body_gets_the_same_413_as_content_length(monkeypatch):
    # Uncaught, the limit reached _serve_connection's connection-level handler and the socket
    # closed with no response at all, while the Content-Length form got a 413.
    monkeypatch.setattr(sr, "_BODY_LIMIT", 1024)

    async def scenario():
        a = await FakeLlama("a").start()
        router = await _router(a)
        try:
            payload = b"x" * 4096
            async with httpx.AsyncClient(timeout = 10) as client:

                async def _chunks():
                    yield payload

                chunked = await client.post(
                    f"{router.base_url}/v1/chat/completions", content = _chunks()
                )
                assert chunked.status_code == 413
                sized = await client.post(f"{router.base_url}/v1/chat/completions", content = payload)
                assert sized.status_code == 413
                assert a.generation_count == 0
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


def test_a_client_that_leaves_while_queued_gives_its_slot_back():
    # dispatch waits in the admission queue, and for a non-streaming call llama-server may
    # send no headers until the generation is finished. Watching for the disconnect only
    # after dispatch returned left the request queued, forwarded and prefilled for nobody.
    async def scenario():
        hold = asyncio.Event()
        a = await FakeLlama("a", hold = hold).start()
        router = await _router(a, slots = 1)
        try:
            async with httpx.AsyncClient(timeout = 10) as holder:
                first = asyncio.ensure_future(_chat(holder, router.base_url, {"prompt": "x"}))
                await _until(lambda: router.get_backend("a").in_flight == 1)

                # Second caller queues behind it, then goes away before it is admitted.
                leaver = httpx.AsyncClient(timeout = 10)
                queued = asyncio.ensure_future(_chat(leaver, router.base_url, {"prompt": "y"}))
                await _until(lambda: router.get_backend("a").queued == 1)
                queued.cancel()
                try:
                    await queued
                except (asyncio.CancelledError, Exception):
                    pass
                await leaver.aclose()

                await _until(lambda: router.get_backend("a").queued == 0)
                hold.set()
                await first
                await _until(lambda: router.get_backend("a").in_flight == 0)
                # The abandoned request never reached llama-server.
                assert a.generation_count == 1
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


def test_a_peer_that_died_since_its_health_probe_does_not_cost_a_request():
    # The connect failure happens before any response header is written, so the request can
    # still be placed on the surviving node instead of closing the client's connection.
    async def scenario():
        a = await FakeLlama("a").start()
        b = await FakeLlama("b").start()
        router = await _router(a, b)
        try:
            key_on_b = next(k for k in (f"t{i}" for i in range(200)) if router.pick(k).name == "b")
            await b.stop()  # gone, but still healthy as far as the last probe knows
            assert router.get_backend("b").healthy
            async with httpx.AsyncClient(timeout = 10) as client:
                frames = await _chat(
                    client, router.base_url, {"prompt": "x", CONVERSATION_FIELD: key_on_b}
                )
                assert frames == ["a-0", "a-1", "a-2", "a-3", "[DONE]"]
            assert not router.get_backend("b").healthy
            assert router.status()["retried_elsewhere"] == 1
            assert router.get_backend("b").in_flight == 0
        finally:
            await router.stop()
            await a.stop()

    run(scenario())


def test_every_backend_being_gone_still_ends_the_request():
    # The retry is bounded by the backend count: each failure takes one out of rotation.
    async def scenario():
        a = await FakeLlama("a").start()
        router = await _router(a)
        try:
            await a.stop()
            async with httpx.AsyncClient(timeout = 10) as client:
                with pytest.raises(httpx.HTTPError):
                    await _chat(client, router.base_url, {"prompt": "x"})
            assert not router.get_backend("a").healthy
            assert router.get_backend("a").in_flight == 0
        finally:
            await router.stop()

    run(scenario())


def test_a_replica_that_closes_before_headers_is_retried_on_the_primary():
    """A pre-header disconnect is a connect failure in every way that matters.

    A replica that closes a pooled connection after ACCEPTING the request and before returning
    headers is the ordinary shutdown and crash race on this pair -- it is why the relaunch
    supervisor exists at all. httpx raises `RemoteProtocolError` for it, which fell into the
    generic `HTTPError` branch: the backend was marked neither down nor unreachable, so
    `dispatch` could not fail over to the healthy primary and sticky routing kept choosing the
    same dead peer until the health loop caught up. No client bytes have been written at that
    point, so the request is safely retryable.
    """

    async def scenario():
        a = await FakeLlama("a").start()

        # A backend that accepts, reads the request and closes without writing a response.
        # A real llama-server dying between accept() and its first write looks exactly so.
        accepted = []

        async def _hang_up(reader, writer):
            accepted.append(1)
            try:
                await reader.read(1)
            except Exception:
                pass
            writer.close()

        dead = await asyncio.start_server(_hang_up, "127.0.0.1", 0)
        port = dead.sockets[0].getsockname()[1]

        router = SparkRouter(health_interval = 3600.0)
        router.add_backend("a", "127.0.0.1", a.port, 4, primary = True)
        router.add_backend("b", "127.0.0.1", port, 4)
        await router.start()
        try:
            # The peer answers its health probe (it accepts), so the router believes it.
            router.get_backend("b").healthy = True
            key_on_b = next(k for k in (f"t{i}" for i in range(200)) if router.pick(k).name == "b")
            async with httpx.AsyncClient(timeout = 10) as client:
                frames = await _chat(
                    client, router.base_url, {"prompt": "x", CONVERSATION_FIELD: key_on_b}
                )
                assert frames == ["a-0", "a-1", "a-2", "a-3", "[DONE]"]
            assert accepted, "the request really did reach the backend that hung up"
            assert not router.get_backend("b").healthy, "a pre-header disconnect marks it down"
            assert router.status()["retried_elsewhere"] == 1
            assert router.get_backend("b").in_flight == 0
        finally:
            await router.stop()
            dead.close()
            await dead.wait_closed()
            await a.stop()

    run(scenario())
