# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

_backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

inference_route = pytest.importorskip("routes.inference", reason = "inference stack not installed")
rag_embeddings = pytest.importorskip("core.rag.embeddings", reason = "rag stack not installed")
rag_config = pytest.importorskip("core.rag.config", reason = "rag stack not installed")

MODEL = "unsloth/bge-small-en-v1.5"


class _Request:
    def __init__(self, body):
        self._body = body
        self.method = "POST"
        self.url = SimpleNamespace(path = "/v1/embeddings")
        self.state = SimpleNamespace(skip_api_monitor = True)
        self.scope = {"type": "http", "path": "/v1/embeddings"}

    async def json(self):
        return self._body

    async def is_disconnected(self):
        return False


def _vectors(texts, **_):
    return np.array([[1.0, 0.0]] * len(texts), dtype = np.float32)


@pytest.fixture
def studio_embedder(monkeypatch):
    async def passthrough(request, current_subject, **_kwargs):
        return await request.json()

    def no_proxy():
        raise AssertionError("the llama-server proxy must not run")

    monkeypatch.setattr(inference_route, "_should_validate_before_switch", lambda: False)
    monkeypatch.setattr(inference_route, "_auto_switch_from_request_body", passthrough)
    monkeypatch.setattr(inference_route, "_cancelable_nonstreaming_client", no_proxy)
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: MODEL)
    monkeypatch.setattr(rag_embeddings, "encode", _vectors)
    monkeypatch.setattr(rag_embeddings, "token_counter", lambda model_name = None: len)
    monkeypatch.setattr(rag_embeddings, "max_tokens", lambda model_name = None: None)
    monkeypatch.setattr(rag_embeddings, "dim", lambda model_name = None: 2)
    return monkeypatch


def _call(body):
    response = asyncio.run(inference_route.openai_embeddings(_Request(body), "tester"))
    assert response.status_code == 200
    import json

    return json.loads(response.body)


def _http_error(body):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_Request(body), "tester"))
    return exc.value


def test_nothing_loaded_serves_from_the_studio_embedder(studio_embedder):
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    payload = _call({"input": ["alpha", "beta"], "model": "text-embedding-3-small"})
    assert payload["object"] == "list"
    assert payload["model"] == MODEL
    assert [row["index"] for row in payload["data"]] == [0, 1]
    assert payload["data"][0]["embedding"] == [1.0, 0.0]
    assert payload["usage"] == {"prompt_tokens": 9, "total_tokens": 9}


def test_chat_model_loaded_serves_from_the_studio_embedder(studio_embedder):
    started = []
    studio_embedder.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = True, is_embedding_gguf = False),
    )
    studio_embedder.setattr(
        inference_route, "_direct_llama_request_started", lambda: started.append(1)
    )
    payload = _call({"input": "alpha"})
    assert payload["model"] == MODEL
    assert started == []


def test_resident_embedding_gguf_still_uses_the_proxy(studio_embedder):
    import httpx

    class _Client:
        async def post(self, *_args, **_kwargs):
            return httpx.Response(200, json = {"data": [{"embedding": [0.5]}]})

        async def aclose(self):
            return None

    def boom(*_args, **_kwargs):
        raise AssertionError("the studio embedder must not run")

    studio_embedder.setattr(rag_embeddings, "encode", boom)
    studio_embedder.setattr(inference_route, "_cancelable_nonstreaming_client", _Client)
    studio_embedder.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            is_embedding_gguf = True,
            base_url = "http://llama.test",
            context_length = 512,
            model_identifier = "org/E-GGUF",
        ),
    )
    payload = _call({"input": "alpha", "model": "org/E-GGUF"})
    assert payload == {"data": [{"embedding": [0.5]}]}


def test_base64_encoding_format(studio_embedder):
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    payload = _call({"input": "alpha", "encoding_format": "base64"})
    raw = base64.b64decode(payload["data"][0]["embedding"])
    assert np.frombuffer(raw, dtype = np.float32).tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    "body",
    [
        {"input": ""},
        {"input": []},
        {"input": [[1, 2, 3]]},
        {"input": ["alpha", ""]},
        {"input": "alpha", "encoding_format": "int8"},
        {"input": "alpha", "dimensions": 999},
    ],
)
def test_invalid_requests_are_400(studio_embedder, body):
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    assert _http_error(body).status_code == 400


def test_over_length_input_is_rejected_not_truncated(studio_embedder):
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    studio_embedder.setattr(rag_embeddings, "max_tokens", lambda model_name = None: 3)
    error = _http_error({"input": ["ab", "abcd"]})
    assert error.status_code == 400
    assert "3-token limit" in error.detail


def test_embedder_failure_is_502(studio_embedder):
    def boom(*_args, **_kwargs):
        raise RuntimeError("llama-server embedder POST /v1/embeddings -> 500")

    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    studio_embedder.setattr(rag_embeddings, "encode", boom)
    assert _http_error({"input": "alpha"}).status_code == 502


def test_studio_embedder_requests_are_admission_limited(studio_embedder):
    lock = threading.Lock()
    gate = threading.Event()
    active = {"now": 0, "peak": 0}

    def encode(texts, **_kwargs):
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        gate.wait(timeout = 5)
        with lock:
            active["now"] -= 1
        return _vectors(texts)

    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    studio_embedder.setattr(rag_embeddings, "encode", encode)

    async def run():
        tasks = [
            asyncio.create_task(inference_route.openai_embeddings(_Request({"input": "x"}), "t"))
            for _ in range(inference_route._STUDIO_EMBED_CONCURRENCY + 2)
        ]
        for _ in range(500):
            await asyncio.sleep(0.01)
            if active["now"] == inference_route._STUDIO_EMBED_CONCURRENCY:
                break
        assert active["now"] == inference_route._STUDIO_EMBED_CONCURRENCY
        gate.set()
        responses = await asyncio.gather(*tasks)
        assert all(response.status_code == 200 for response in responses)

    asyncio.run(run())
    assert active["peak"] == inference_route._STUDIO_EMBED_CONCURRENCY


def test_studio_fallback_untracks_the_request_from_the_llama_slot(studio_embedder):
    # /v1/embeddings is an _INFERENCE_SUFFIXES path and is NOT in _NON_LLM_SLOT_SUFFIXES, so a
    # 2xx that reaches llama_keepwarm._finish claims the llama slot and clears preview ownership.
    # The studio embedder never touches the resident GGUF, so it must untrack first -- the same
    # contract the external-provider chat branch follows.
    from core.inference import llama_keepwarm as kw

    studio_embedder.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = True, is_embedding_gguf = False),
    )
    request = _Request({"input": "alpha"})
    before = kw._inflight
    kw._inflight = before + 1
    try:
        response = asyncio.run(inference_route.openai_embeddings(request, "tester"))
        assert response.status_code == 200
        # Set => _finish() skips both _claim_non_preview_slot() and the activity stamp.
        assert request.scope.get(kw._UNTRACKED_SCOPE_KEY) is True
        assert kw._inflight == before
    finally:
        kw._inflight = before


def test_resident_embedding_gguf_still_claims_the_slot(studio_embedder):
    # The proxy path DOES run against the resident GGUF, so it must stay tracked.
    import httpx

    class _Client:
        async def post(self, *_args, **_kwargs):
            return httpx.Response(200, json = {"data": [{"embedding": [0.5]}]})

        async def aclose(self):
            return None

    from core.inference import llama_keepwarm as kw

    studio_embedder.setattr(inference_route, "_cancelable_nonstreaming_client", _Client)
    studio_embedder.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            is_embedding_gguf = True,
            base_url = "http://llama.test",
            context_length = 512,
            model_identifier = "org/E-GGUF",
        ),
    )
    request = _Request({"input": "alpha", "model": "org/E-GGUF"})
    asyncio.run(inference_route.openai_embeddings(request, "tester"))
    assert request.scope.get(kw._UNTRACKED_SCOPE_KEY) is None


def test_context_gauge_is_not_pinned_by_a_batch(studio_embedder):
    # _monitor_openai_chunk's 3rd arg is context_length, and api_monitor divides the batch's
    # summed prompt_tokens by it. Passing the per-text limit for a multi-input request would
    # report 100% context use for a request that used a fraction of it per text.
    seen = []
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    studio_embedder.setattr(rag_embeddings, "max_tokens", lambda model_name = None: 8)
    studio_embedder.setattr(
        inference_route,
        "_monitor_openai_chunk",
        lambda monitor_id, payload, context_length = None, **_: seen.append(context_length),
    )
    request = _Request({"input": ["alpha", "beta", "gamma"]})
    request.state.skip_api_monitor = False
    asyncio.run(inference_route.openai_embeddings(request, "tester"))
    assert seen == [None]

    seen.clear()
    request = _Request({"input": "alpha"})
    request.state.skip_api_monitor = False
    asyncio.run(inference_route.openai_embeddings(request, "tester"))
    assert seen == [8]


def test_cancelled_requests_do_not_leak_admission_permits(studio_embedder):
    # to_thread cannot cancel the worker thread. If the permit were released when the awaiting
    # task is cancelled, a client that disconnects (or a shutdown) would let the next batch in
    # while the old threads are still embedding, so the concurrency cap would stop holding.
    lock = threading.Lock()
    gate = threading.Event()
    active = {"now": 0, "peak": 0}

    def encode(texts, **_kwargs):
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        gate.wait(timeout = 5)
        with lock:
            active["now"] -= 1
        return _vectors(texts)

    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    studio_embedder.setattr(rag_embeddings, "encode", encode)
    cap = inference_route._STUDIO_EMBED_CONCURRENCY

    async def run():
        first = [
            asyncio.create_task(inference_route.openai_embeddings(_Request({"input": "x"}), "t"))
            for _ in range(cap)
        ]
        for _ in range(500):
            await asyncio.sleep(0.01)
            if active["now"] == cap:
                break
        assert active["now"] == cap

        # Every in-flight request goes away, but its thread is still inside encode().
        for task in first:
            task.cancel()
        await asyncio.gather(*first, return_exceptions = True)

        second = [
            asyncio.create_task(inference_route.openai_embeddings(_Request({"input": "y"}), "t"))
            for _ in range(cap)
        ]
        await asyncio.sleep(0.3)
        # The permits are still held by the running threads, so nothing new started.
        assert active["peak"] == cap, f"admission cap exceeded: peak={active['peak']}"
        gate.set()
        await asyncio.gather(*second, return_exceptions = True)

    asyncio.run(run())
    assert active["peak"] == cap
