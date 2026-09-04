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
IDENTITY = f"sentence-transformers:{MODEL}"


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

    def _encode_with_identity(
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        # Delegates to whatever encode the test installed, so a test that makes encode
        # blow up or block still drives the real route.
        return rag_embeddings.encode(texts, model_name = model_name, normalize = normalize), IDENTITY

    monkeypatch.setattr(rag_embeddings, "encode_with_identity", _encode_with_identity)
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
    assert payload["model"] == IDENTITY
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
    assert payload["model"] == IDENTITY
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


def test_reported_model_follows_the_backend_that_made_the_vectors(studio_embedder):
    # _SentenceTransformersBackend.encode swaps the process to llama-server when ST encode
    # fails, and that is a different embedding space. The response must name the space the
    # vectors are actually in, or a client stores two spaces under one label.
    llama_identity = f"llama-server:{MODEL}:unsloth/bge-small-en-v1.5-GGUF"
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    studio_embedder.setattr(
        rag_embeddings,
        "encode_with_identity",
        lambda texts, **_kwargs: (_vectors(texts), llama_identity),
    )
    payload = _call({"input": "alpha"})
    assert payload["model"] == llama_identity


def test_embedding_helpers_are_pinned_to_the_captured_model(studio_embedder):
    # A Settings change while the request queues must not mix one model's limit/dim with
    # another model's vectors: every helper is called with the model captured up front.
    seen = {"max_tokens": [], "token_counter": [], "dim": [], "encode": []}
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    studio_embedder.setattr(
        rag_embeddings,
        "max_tokens",
        lambda model_name = None: (seen["max_tokens"].append(model_name), None)[1],
    )
    studio_embedder.setattr(
        rag_embeddings,
        "token_counter",
        lambda model_name = None: (seen["token_counter"].append(model_name), len)[1],
    )
    studio_embedder.setattr(
        rag_embeddings,
        "dim",
        lambda model_name = None: (seen["dim"].append(model_name), 2)[1],
    )
    studio_embedder.setattr(
        rag_embeddings,
        "encode_with_identity",
        lambda texts, **kw: (
            seen["encode"].append(kw.get("model_name")),
            (_vectors(texts), IDENTITY),
        )[1],
    )
    _call({"input": "alpha", "dimensions": 2})
    assert seen == {
        "max_tokens": [MODEL],
        "token_counter": [MODEL],
        "dim": [MODEL],
        "encode": [MODEL],
    }


def test_studio_fallback_releases_the_preview_busy_guard(studio_embedder):
    # The auto-switch hook admits every /v1/embeddings request before the route decides how to
    # serve it. load_model_for_preview reads the admitted tally, not _inflight, so without
    # clearing it a slow studio encode keeps rejecting preview swaps with 503.
    from core.inference import llama_keepwarm as kw

    studio_embedder.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = True, is_embedding_gguf = False),
    )
    request = _Request({"input": "alpha"})
    # Relative to whatever the tally already holds: it is module state shared with the rest of
    # the suite, so an absolute count is only right when this file runs alone.
    before = kw.other_admitted_inference_count()
    kw.note_admitted_inference(request.scope)
    assert kw.other_admitted_inference_count() == before + 1
    assert request.scope.get(kw._ADMITTED_SCOPE_KEY) is True
    try:
        asyncio.run(inference_route.openai_embeddings(request, "tester"))
        assert kw.other_admitted_inference_count() == before
        # Popped, so the middleware's finally cannot decrement the tally a second time.
        assert request.scope.get(kw._ADMITTED_SCOPE_KEY) is None
    finally:
        kw._admitted_inference = before


def test_cancelled_request_closes_its_monitor_row(studio_embedder):
    # api_monitor.start() runs before admission. Running rows are excluded from retention
    # trimming, so a request cancelled while queued (or mid-encode) must close its own row.
    closed = []
    gate = threading.Event()
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    studio_embedder.setattr(inference_route.api_monitor, "start", lambda **_kwargs: "entry-1")
    studio_embedder.setattr(
        inference_route.api_monitor,
        "finish",
        lambda entry_id, *args, **_kw: closed.append((entry_id, args[0] if args else None)),
    )
    studio_embedder.setattr(
        rag_embeddings,
        "encode_with_identity",
        lambda texts, **_kw: (gate.wait(timeout = 5), (_vectors(texts), IDENTITY))[1],
    )

    async def run():
        request = _Request({"input": "alpha"})
        request.state.skip_api_monitor = False
        task = asyncio.create_task(inference_route.openai_embeddings(request, "tester"))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()

    asyncio.run(run())
    assert closed == [("entry-1", "cancelled")]


def _identity_names(monkeypatch):
    monkeypatch.setattr(
        rag_config, "effective_gguf_repo_for_embedding_model", lambda model: f"{model}-GGUF"
    )
    monkeypatch.setattr(rag_embeddings, "embedding_identity", lambda model_name = None: IDENTITY)


@pytest.mark.parametrize("requested", [MODEL, f"{MODEL}-GGUF", IDENTITY, MODEL.upper()])
def test_naming_the_configured_embedder_skips_the_chat_slot_check(studio_embedder, requested):
    from fastapi import HTTPException

    async def reject(request, current_subject, **_kwargs):
        raise HTTPException(status_code = 404, detail = "model_not_found")

    _identity_names(studio_embedder)
    studio_embedder.setattr(inference_route, "_auto_switch_from_request_body", reject)
    studio_embedder.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = True, is_embedding_gguf = False),
    )
    payload = _call({"input": "alpha", "model": requested})
    assert payload["model"] == IDENTITY
    assert payload["data"][0]["embedding"] == [1.0, 0.0]


def test_other_model_names_still_run_auto_switch(studio_embedder):
    seen = []

    async def record(request, current_subject, **_kwargs):
        seen.append(current_subject)
        return await request.json()

    _identity_names(studio_embedder)
    studio_embedder.setattr(inference_route, "_auto_switch_from_request_body", record)
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    _call({"input": "alpha", "model": "text-embedding-3-small"})
    assert seen == ["tester"]


def test_st_max_tokens_leaves_room_for_the_special_tokens(monkeypatch):
    model = SimpleNamespace(
        max_seq_length = 512,
        tokenizer = SimpleNamespace(num_special_tokens_to_add = lambda: 2),
    )
    monkeypatch.setattr(rag_embeddings, "_get", lambda model_name = None: model)
    assert rag_embeddings._SentenceTransformersBackend().max_tokens() == 510


def _gguf(tmp_path, entries):
    import io
    import struct

    buf = io.BytesIO()
    buf.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)))
    for key, vtype, value in entries:
        raw = key.encode()
        buf.write(struct.pack("<Q", len(raw)) + raw + struct.pack("<I", vtype))
        if vtype == 8:
            raw = value.encode()
            buf.write(struct.pack("<Q", len(raw)) + raw)
        elif vtype == 9:
            elem_type, items = value
            buf.write(struct.pack("<IQ", elem_type, len(items)))
            for item in items:
                raw = item.encode()
                buf.write(struct.pack("<Q", len(raw)) + raw)
        elif vtype == 4:
            buf.write(struct.pack("<I", value))
        elif vtype == 10:
            buf.write(struct.pack("<Q", value))
    path = tmp_path / "embedder.gguf"
    path.write_bytes(buf.getvalue())
    return str(path)


def test_gguf_context_length_is_read_from_the_header(tmp_path):
    from core.rag import embed_llama_server

    path = _gguf(
        tmp_path,
        [
            ("general.architecture", 8, "bert"),
            ("tokenizer.ggml.tokens", 9, (8, ["[PAD]", "[CLS]", "hello"])),
            ("llama.context_length", 4, 4096),
            ("bert.context_length", 10, 512),
        ],
    )
    assert embed_llama_server._gguf_context_length(path) == 512
    assert embed_llama_server._gguf_context_length(str(tmp_path / "missing.gguf")) is None


def test_llama_max_tokens_comes_from_the_gguf_minus_its_special_tokens(tmp_path, monkeypatch):
    from core.rag import embed_llama_server

    backend = embed_llama_server.LlamaServerBackend()
    backend._model_path = _gguf(
        tmp_path, [("general.architecture", 8, "bert"), ("bert.context_length", 4, 512)]
    )
    monkeypatch.setattr(backend, "_ensure_ready", lambda model_name = None: None)
    monkeypatch.setattr(backend, "_server_context", lambda: None)
    posts = []

    def post(
        path,
        payload,
        model_name = None,
    ):
        posts.append((path, payload))
        return {"tokens": [101, 102]}

    monkeypatch.setattr(backend, "_post", post)
    assert backend.max_tokens() == 510
    assert backend.max_tokens() == 510
    assert posts == [("/tokenize", {"content": "", "add_special": True})]
    backend._adopt_model_path(backend._model_path, "unsloth/other-GGUF")
    assert backend._max_tokens is None


def test_st_max_tokens_reserves_the_default_prompt(monkeypatch):
    tokenizer = SimpleNamespace(
        num_special_tokens_to_add = lambda: 2,
        encode = lambda text, add_special_tokens = False: text.split(),
    )
    model = SimpleNamespace(
        max_seq_length = 512,
        tokenizer = tokenizer,
        prompts = {"query": "Represent this sentence for retrieval:"},
        default_prompt_name = "query",
    )
    monkeypatch.setattr(rag_embeddings, "_get", lambda model_name = None: model)
    assert rag_embeddings._SentenceTransformersBackend().max_tokens() == 512 - 2 - 5


def test_llama_max_tokens_is_capped_by_the_running_context(tmp_path, monkeypatch):
    from core.rag import embed_llama_server

    backend = embed_llama_server.LlamaServerBackend()
    backend._model_path = _gguf(
        tmp_path, [("general.architecture", 8, "bert"), ("bert.context_length", 4, 512)]
    )
    monkeypatch.setattr(backend, "_ensure_ready", lambda model_name = None: None)
    monkeypatch.setattr(backend, "_server_context", lambda: 256)
    monkeypatch.setattr(backend, "_post", lambda *a, **k: {"tokens": [101, 102]})
    assert backend.max_tokens() == 254


@pytest.mark.parametrize(
    ("body", "switched"),
    [
        ({"input": [""], "model": "org/chat-GGUF"}, False),
        ({"input": ["alpha", 7], "model": "org/chat-GGUF"}, False),
        ({"input": "alpha", "encoding_format": "int8", "model": "org/chat-GGUF"}, False),
        ({"input": [[1, 2, 3]], "model": "org/chat-GGUF"}, True),
    ],
)
def test_invalid_fallback_input_is_rejected_before_the_switch(studio_embedder, body, switched):
    seen = []

    async def record(request, current_subject, **_kwargs):
        seen.append(current_subject)
        return await request.json()

    studio_embedder.setattr(inference_route, "_should_validate_before_switch", lambda: True)
    studio_embedder.setattr(inference_route, "_auto_switch_from_request_body", record)
    studio_embedder.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = True, is_embedding_gguf = False),
    )
    assert _http_error(body).status_code == 400
    assert bool(seen) is switched


def test_local_path_models_are_not_exposed(studio_embedder, tmp_path):
    from core.rag.config import _escape_identity_segment

    model_dir = str(tmp_path / "bge")
    studio_embedder.setattr(rag_config, "effective_embedding_model", lambda: model_dir)
    studio_embedder.setattr(
        rag_config, "effective_gguf_repo_for_embedding_model", lambda model: f"{model}-GGUF"
    )
    identity = f"sentence-transformers:{_escape_identity_segment(model_dir)}"
    studio_embedder.setattr(
        rag_embeddings,
        "encode_with_identity",
        lambda texts, **_kwargs: (_vectors(texts), identity),
    )
    studio_embedder.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    assert _call({"input": "alpha"})["model"] == "sentence-transformers:bge"
    studio_embedder.setattr(rag_embeddings, "max_tokens", lambda model_name = None: 3)
    error = _http_error({"input": "alphabet"})
    assert error.status_code == 400
    assert "bge" in error.detail and str(tmp_path) not in error.detail
