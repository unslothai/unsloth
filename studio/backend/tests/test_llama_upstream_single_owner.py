# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import os
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

pytest.importorskip("fastapi", reason = "inference stack not installed")

import routes.inference as inf_mod  # noqa: E402
from models.inference import ChatCompletionRequest, ChatMessage, ResponsesRequest  # noqa: E402
from state import active_generations  # noqa: E402

_DEAD = "http://127.0.0.1:57953"
_FRESH = "http://127.0.0.1:62933"

_CRASHES = [
    pytest.param(httpx.ConnectError, id = "refused"),
    pytest.param(httpx.RemoteProtocolError, id = "dropped-before-headers"),
]


class _Backend:
    is_loaded = True
    is_vision = False
    context_length = 4096
    model_identifier = "org/M-GGUF"
    supports_reasoning = True
    reasoning_always_on = False

    def __init__(self, *, stays_dead = False):
        self.base_url = _DEAD
        self._auth_headers = {"Authorization": "Bearer before-crash"}
        self._stays_dead = stays_dead
        self.respawn_calls = 0

    def _request_reasoning_kwargs(self, *_args, **_kwargs):
        return None

    def count_chat_tokens(self, *_args, **_kwargs):
        return 2

    def _maybe_recover_from_mtp_crash(self, _exc):
        return False

    def _respawn_if_dead(self):
        self.respawn_calls += 1
        if not self._stays_dead:
            self.base_url = _FRESH
            self._auth_headers = {"Authorization": "Bearer after-crash"}
        return True


def _sse(*events):
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(
        200, content = content.encode(), headers = {"content-type": "text/event-stream"}
    )


def _reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    path = request.url.path
    if path == "/v1/embeddings":
        return httpx.Response(200, json = {"data": [{"embedding": [0.33, 0.2]}]})
    if path == "/v1/completions":
        if body.get("stream"):
            return _sse({"choices": [{"text": "33"}]})
        return httpx.Response(200, json = {"id": "cmpl-1", "choices": [{"text": "33"}]})
    if body.get("stream"):
        return _sse({"choices": [{"delta": {"content": "33"}}]})
    return httpx.Response(
        200,
        json = {
            "id": "chatcmpl-1",
            "choices": [
                {"message": {"role": "assistant", "content": "33"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
    )


@pytest.fixture
def llama(monkeypatch):
    active_generations.reset_for_tests()
    state = SimpleNamespace(crash = httpx.ConnectError, calls = [], backend = _Backend(), reply_delay = 0)

    async def handler(request):
        state.calls.append((str(request.url), request.headers.get("authorization")))
        if str(request.url).startswith(_DEAD):
            raise state.crash("llama-server is gone", request = request)
        if state.reply_delay:
            await asyncio.sleep(state.reply_delay)
        return _reply(request)

    transport = httpx.MockTransport(handler)
    state.transport = transport
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *a, **kw: real_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: state.backend)
    monkeypatch.setattr(inf_mod, "_automatic_model_load_may_run", lambda: False)
    # Settings I/O must not consume the recovery tests' deadlines before dispatch.
    monkeypatch.setattr(inf_mod, "configured_first_token_timeout_s", lambda: 600.0)
    monkeypatch.setattr("utils.inference.inference_config._saved_sampling", lambda _model: {})

    async def _no_auto_switch(request, current_subject, **_kwargs):
        return await request.json()

    monkeypatch.setattr(inf_mod, "_auto_switch_from_request_body", _no_auto_switch)
    yield state
    active_generations.reset_for_tests()


class _Request:
    def __init__(
        self,
        path,
        body = None,
    ):
        self._body = body
        self.method = "POST"
        self.url = SimpleNamespace(path = path)
        self.state = SimpleNamespace(skip_api_monitor = True)

    async def json(self):
        return self._body

    async def is_disconnected(self):
        return False


async def _drain(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


async def _completions(stream):
    body = {"prompt": "hi", "model": "org/M-GGUF", "max_tokens": 8, "stream": stream}
    response = await inf_mod.openai_completions(_Request("/v1/completions", body), "tester")
    return await _drain(response) if stream else response.body.decode()


async def _embeddings():
    body = {"input": "hi", "model": "org/M-GGUF"}
    response = await inf_mod.openai_embeddings(_Request("/v1/embeddings", body), "tester")
    return response.body.decode()


async def _responses_stream():
    payload = ResponsesRequest(input = "hi", stream = True, model = "org/M-GGUF")
    messages = [ChatMessage(role = "user", content = "hi")]
    response = await inf_mod._responses_stream(payload, messages, _Request("/v1/responses"))
    return await _drain(response)


_MESSAGES_ARGS = ([{"role": "user", "content": "hi"}], [], 0.7, 0.95, 20, 16, "msg_1", "m")


async def _messages_stream():
    backend = inf_mod.get_llama_cpp_backend()
    response = await inf_mod._anthropic_passthrough_stream(
        _Request("/v1/messages"), threading.Event(), backend, *_MESSAGES_ARGS
    )
    return await _drain(response)


async def _messages():
    response = await inf_mod._anthropic_passthrough_non_streaming(
        inf_mod.get_llama_cpp_backend(),
        *_MESSAGES_ARGS,
        request = _Request("/v1/messages"),
        cancel_event = threading.Event(),
    )
    return response.body.decode()


async def _chat_stream():
    cancel = threading.Event()
    tracker = inf_mod._TrackedCancel(cancel, model = "org/M-GGUF", kind = "chat")
    tracker.__enter__()
    lease = SimpleNamespace(released = False)
    lease.release = lambda: setattr(lease, "released", True)
    try:
        response = await inf_mod._openai_passthrough_stream_admitted(
            _Request("/v1/chat/completions"),
            cancel,
            inf_mod.get_llama_cpp_backend(),
            ChatCompletionRequest(messages = [ChatMessage(role = "user", content = "hi")], stream = True),
            "org/M-GGUF",
            "chatcmpl-1",
            admission_lease = lease,
            tracker = tracker,
        )
        return await _drain(response)
    finally:
        assert lease.released


async def _chat_nonstream():
    return await inf_mod._openai_passthrough_non_streaming_upstream(
        inf_mod.get_llama_cpp_backend(),
        ChatCompletionRequest(messages = [ChatMessage(role = "user", content = "hi")]),
        "org/M-GGUF",
        request = _Request("/v1/chat/completions"),
        cancel_event = threading.Event(),
    )


_ROUTES = [
    pytest.param(lambda: _completions(stream = True), "/v1/completions", id = "completions-stream"),
    pytest.param(lambda: _completions(stream = False), "/v1/completions", id = "completions"),
    pytest.param(_embeddings, "/v1/embeddings", id = "embeddings"),
    pytest.param(_responses_stream, "/v1/chat/completions", id = "responses-stream"),
    pytest.param(_messages_stream, "/v1/chat/completions", id = "messages-stream"),
    pytest.param(_messages, "/v1/chat/completions", id = "messages"),
    pytest.param(_chat_stream, "/v1/chat/completions", id = "chat-stream"),
]


@pytest.mark.parametrize("crash", _CRASHES)
@pytest.mark.parametrize("call,path", _ROUTES)
def test_every_route_recovers_from_a_crashed_llama_server(llama, call, path, crash):
    llama.crash = crash

    body = asyncio.run(call())

    assert "33" in body
    assert llama.backend.respawn_calls == 1
    assert llama.calls == [
        (f"{_DEAD}{path}", "Bearer before-crash"),
        (f"{_FRESH}{path}", "Bearer after-crash"),
    ]
    assert active_generations.count() == 0


def _upstream(**kwargs):
    return inf_mod._LlamaUpstream(_Backend(**kwargs), "/v1/chat/completions")


def test_a_request_respawns_at_most_once(llama):
    llama.backend = _Backend(stays_dead = True)
    upstream = inf_mod._LlamaUpstream(llama.backend, "/v1/chat/completions")

    async def run():
        async with upstream:
            for _ in range(2):  # the passthrough's nudge retry posts again on the same request
                with pytest.raises(httpx.ConnectError):
                    await upstream.post({}, timeout = 5)

    asyncio.run(run())

    assert llama.backend.respawn_calls == 1
    assert len(llama.calls) == 3


@pytest.mark.parametrize(
    "exc",
    [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), httpx.PoolTimeout("slow")],
)
def test_a_slow_server_is_never_respawned(exc):
    upstream = _upstream()

    assert not asyncio.run(upstream.retarget_after_respawn(exc))
    assert upstream._backend.respawn_calls == 0


class _ClosedClient:
    async def post(self, *_args, **_kwargs):
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    async def aclose(self):
        pass


def test_a_post_on_a_client_the_cancel_closed_reports_the_cancel(monkeypatch):
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", _ClosedClient)
    cancel = threading.Event()
    cancel.set()
    upstream = inf_mod._LlamaUpstream(_Backend(), "/v1/embeddings", cancel_event = cancel)

    async def run():
        async with upstream:
            await upstream.post({}, timeout = 5, path = "/tokenize")

    with pytest.raises(inf_mod._NonStreamingRequestCancelled):
        asyncio.run(run())


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: _completions(False), id = "completions"),
        pytest.param(_embeddings, id = "embeddings"),
        pytest.param(_messages, id = "messages"),
        pytest.param(_chat_nonstream, id = "chat"),
    ],
)
def test_cancellation_during_watcher_cleanup_closes_client_and_propagates(llama, monkeypatch, call):
    llama.backend.base_url = _FRESH
    client = httpx.AsyncClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    stop_watcher = inf_mod._stop_local_disconnect_cancel_watcher
    stopping = threading.Event()

    async def blocked_stop(watcher):
        await stop_watcher(watcher)
        stopping.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(inf_mod, "_stop_local_disconnect_cancel_watcher", blocked_stop)

    async def run():
        task = asyncio.create_task(call())
        try:
            await _until(stopping.is_set)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)
        assert client.is_closed
        assert active_generations.count() == 0
        assert inf_mod._direct_llama_inflight == 0

    asyncio.run(run())


@pytest.mark.parametrize("restart_after", ["embeddings", "first-tokenize"])
def test_embedding_truncation_keeps_url_and_key_together_after_another_request_respawns(
    llama, monkeypatch, restart_after
):
    backend = llama.backend
    calls = []
    embedded = []
    tokenized = []

    async def handler(request):
        url, key = str(request.url), request.headers.get("authorization")
        calls.append((url, key))
        if url.startswith(_DEAD) and backend.base_url == _FRESH:
            raise httpx.ConnectError("the old server exited", request = request)
        if key != backend._auth_headers["Authorization"]:
            return httpx.Response(401, json = {"error": "invalid API key"})
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            tokenized.append(body["add_special"])
            tokens = [11, 12, 13, 14]
            if body["add_special"]:
                tokens = [101, *tokens, 102]
                if restart_after == "first-tokenize":
                    backend._respawn_if_dead()
            return httpx.Response(200, json = {"tokens": tokens})
        embedded.append(body["input"])
        if isinstance(body["input"], str):
            if restart_after == "embeddings":
                backend._respawn_if_dead()
            return httpx.Response(
                400,
                json = {
                    "error": {
                        "message": "input (6 tokens) is larger than the max context size (4 tokens). skipping"
                    }
                },
            )
        return _reply(request)

    monkeypatch.setattr(llama.transport, "handler", handler)
    request = _Request("/v1/embeddings", {"input": "alpha beta gamma delta", "model": "org/M-GGUF"})
    response = asyncio.run(inf_mod.openai_embeddings(request, "tester"))

    assert response.status_code == 200
    assert embedded == ["alpha beta gamma delta", [[101, 11, 12, 102]]]
    assert tokenized == [True, False]
    assert backend.respawn_calls == 2  # The other request reloads; this request joins recovery.
    assert all(
        key == ("Bearer before-crash" if url.startswith(_DEAD) else "Bearer after-crash")
        for url, key in calls
    )
    assert calls[-1] == (f"{_FRESH}/v1/embeddings", "Bearer after-crash")
    assert active_generations.count() == 0


class _BlockedRespawnBackend(_Backend):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def _respawn_if_dead(self):
        self.started.set()
        try:
            if not self.release.wait(timeout = 5):
                raise RuntimeError("test did not release the reload")
            return super()._respawn_if_dead()
        finally:
            self.finished.set()


async def _until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout = 1)


_STREAM_ROUTES = [_ROUTES[i] for i in (0, 3, 4, 6)]


@pytest.mark.parametrize("call,path", _STREAM_ROUTES)
@pytest.mark.parametrize("reply_delay", [0, 0.1], ids = ["immediate-retry", "slow-retry"])
def test_streams_emit_keepalives_before_a_slow_reload_finishes(
    llama, monkeypatch, call, path, reply_delay
):
    backend = llama.backend = _BlockedRespawnBackend()
    llama.reply_delay = reply_delay
    monkeypatch.setattr(inf_mod, "_OPENAI_PASSTHROUGH_PENDING_RESPONSE_KEEPALIVE_S", 0.02)
    saw_keepalive = threading.Event()

    async def drain(response):
        chunks = []
        async for chunk in response.body_iterator:
            chunk = chunk.decode() if isinstance(chunk, bytes) else chunk
            chunks.append(chunk)
            if (
                not saw_keepalive.is_set()
                and backend.started.is_set()
                and chunk == inf_mod._OPENAI_PASSTHROUGH_SSE_KEEPALIVE
            ):
                assert not backend.finished.is_set()
                saw_keepalive.set()
        return "".join(chunks)

    monkeypatch.setattr(sys.modules[__name__], "_drain", drain)

    async def run():
        task = asyncio.create_task(call())
        try:
            await _until(saw_keepalive.is_set)
        finally:
            backend.release.set()
            body = await task
        assert "33" in body

    asyncio.run(run())
    assert llama.calls == [
        (f"{_DEAD}{path}", "Bearer before-crash"),
        (f"{_FRESH}{path}", "Bearer after-crash"),
    ]
    assert active_generations.count() == 0


@pytest.mark.parametrize("call,path", _ROUTES[:5] + [_ROUTES[6]])
@pytest.mark.parametrize("cancel_kind", ["stop", "disconnect", "task"])
def test_cancellation_releases_request_before_reload_finishes(
    llama, monkeypatch, call, path, cancel_kind
):
    backend = llama.backend = _BlockedRespawnBackend()
    disconnected = False
    finishes = []
    failures = []

    async def is_disconnected(_request):
        return disconnected

    monkeypatch.setattr(_Request, "is_disconnected", is_disconnected)
    monkeypatch.setattr(
        inf_mod.api_monitor, "finish", lambda *args, **kwargs: finishes.append((args, kwargs))
    )
    monkeypatch.setattr(
        inf_mod.api_monitor, "fail", lambda *args, **kwargs: failures.append((args, kwargs))
    )

    async def run():
        nonlocal disconnected
        task = asyncio.create_task(call())
        try:
            await _until(backend.started.is_set)
            assert active_generations.count() > 0
            if cancel_kind == "stop":
                active_generations.cancel_all()
            elif cancel_kind == "disconnect":
                disconnected = True
            else:
                task.cancel()
            await _until(task.done)
            assert not backend.finished.is_set()
            assert active_generations.count() == 0
            assert inf_mod._direct_llama_inflight == 0
            assert not failures
            # Messages' bare stream helper leaves monitor completion to its caller.
            if call is not _messages_stream:
                assert any(args[1:] == ("cancelled",) for args, _kwargs in finishes)
            try:
                task.result()
            except inf_mod.HTTPException as exc:
                assert exc.status_code == 499
            except asyncio.CancelledError:
                assert cancel_kind == "task"
        finally:
            backend.release.set()
            await asyncio.gather(task, return_exceptions = True)
            await _until(backend.finished.is_set)
            await _until(lambda: not inf_mod._LLAMA_RESPAWN_TASKS)

    asyncio.run(run())
    assert llama.calls == [(f"{_DEAD}{path}", "Bearer before-crash")]
    assert backend.respawn_calls == 1


@pytest.mark.parametrize("stream", [False, True])
def test_cancel_at_reload_completion_does_not_send_a_retry(llama, monkeypatch, stream):
    cancel = threading.Event()
    respawn = llama.backend._respawn_if_dead

    def finish_and_cancel():
        recovered = respawn()
        cancel.set()
        return recovered

    monkeypatch.setattr(llama.backend, "_respawn_if_dead", finish_and_cancel)

    async def run():
        upstream = inf_mod._LlamaUpstream(
            llama.backend, "/v1/chat/completions", cancel_event = cancel
        )
        async with upstream:
            if stream:
                assert await upstream.send_stream(upstream.client, {}, cancel, None) is None
            else:
                with pytest.raises(inf_mod._NonStreamingRequestCancelled):
                    await upstream.post({}, timeout = 5)

    asyncio.run(run())
    assert len(llama.calls) == 1
    assert llama.backend.respawn_calls == 1
