# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Numeric text must not overcommit the shared KV pool through a character estimate."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.inference.llama_admission import LlamaAdmissionQueue
import routes.inference as inference


def _backend(count):
    return SimpleNamespace(
        base_url = "test-token-count",
        context_length = 30000,
        effective_parallel_slots = 3,
        count_chat_tokens = Mock(return_value = count),
        _request_reasoning_kwargs = Mock(return_value = {"enable_thinking": False}),
    )


def _payload():
    return SimpleNamespace(
        messages = [{"role": "user", "content": " ".join(map(str, range(1000, 5500)))}],
        max_tokens = 1024,
        enable_thinking = False,
        reasoning_effort = None,
        preserve_thinking = None,
    )


def test_numeric_prompts_wait_for_room_but_short_prompts_still_overlap(monkeypatch):
    async def scenario(count):
        queue = LlamaAdmissionQueue("counted")
        monkeypatch.setattr(inference, "get_llama_admission_queue", lambda _: queue)
        backend = _backend(count)
        payload = _payload()
        reservations = []
        for _ in range(3):
            reservation, _ = await inference._reserve_counted_gguf_chat(
                request = None,
                llama_backend = backend,
                payload = payload,
                messages = payload.messages,
            )
            reservations.append(reservation)
        leases = [r.lease_nowait() for r in reservations]
        if count == 22560:
            assert leases[0] is not None
            assert leases[1:] == [None, None]
            assert queue.snapshot().committed == 23584
            leases[0].release()
            await asyncio.sleep(0)
            assert reservations[1].lease_nowait() is not None
        else:
            assert all(lease is not None for lease in leases)
        assert backend.count_chat_tokens.call_count == 3

    asyncio.run(scenario(22560))
    asyncio.run(scenario(100))


def test_counts_prepared_messages_and_tools_with_generation_options():
    backend = _backend(22560)
    payload = _payload()
    prepared = [{"role": "system", "content": "Injected system instructions"}] + payload.messages
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    count = inference._count_gguf_admission_prompt(backend, payload, prepared, tools)
    assert count == 22560
    args, kwargs = backend.count_chat_tokens.call_args
    assert args[0] == prepared
    assert kwargs == dict(
        tools = tools,
        strict = True,
        prefer_native = True,
        chat_template_kwargs = {"enable_thinking": False},
        continue_final_message = False,
    )
    assert (
        inference._openai_llama_admission_tokens(
            payload,
            budget = 30000,
            capacity = 3,
            injected_tools = tools,
            counted_prompt_tokens = count,
        )
        == 23584
    )


@pytest.mark.parametrize("result", [0, -1, None, True, "22560"])
def test_invalid_counts_reserve_the_entire_pool(result):
    payload = _payload()
    assert (
        inference._count_gguf_admission_prompt(_backend(result), payload, payload.messages) == 30000
    )


def test_tokenizer_failure_reserves_the_entire_pool():
    backend = _backend(0)
    backend.count_chat_tokens.side_effect = RuntimeError("tokenizer unavailable")
    payload = _payload()
    assert inference._count_gguf_admission_prompt(backend, payload, payload.messages) == 30000


def test_tool_recost_keeps_the_model_count(monkeypatch):
    async def scenario():
        queue = LlamaAdmissionQueue("recost")
        monkeypatch.setattr(inference, "get_llama_admission_queue", lambda _: queue)
        backend = _backend(22560)
        payload = _payload()
        reservation, _ = await inference._reserve_counted_gguf_chat(
            request = None,
            llama_backend = backend,
            payload = payload,
            messages = payload.messages,
            tool_loop = True,
        )
        assert reservation.lease_nowait() is not None
        inference._openai_llama_admission_recost(
            reservation,
            payload.messages,
            request = None,
            llama_backend = backend,
            payload = payload,
            count_prepared_prompt = True,
        )
        assert queue.snapshot().committed == 23584

    asyncio.run(scenario())


@pytest.mark.parametrize("failed_path", ["legacy", "native", "unsupported", "invalid", "timeout"])
def test_exact_count_endpoint_failover_preserves_short_chat_concurrency(monkeypatch, failed_path):
    import httpx
    from types import MethodType
    from core.inference.llama_cpp import LlamaCppBackend

    calls = []

    def respond(request):
        path = request.url.path
        calls.append(path)
        if path == "/v1/chat/completions/input_tokens":
            if failed_path == "timeout":
                raise httpx.ReadTimeout("count timed out", request = request)
            if failed_path in ("native", "unsupported"):
                return httpx.Response(404 if failed_path == "unsupported" else 503)
            if failed_path == "invalid":
                return httpx.Response(200, json = {"input_tokens": True})
            return httpx.Response(200, json = {"input_tokens": 20})
        if failed_path == "legacy":
            return httpx.Response(503)
        if path == "/apply-template":
            return httpx.Response(200, json = {"prompt": "rendered prompt"})
        assert path == "/tokenize"
        return httpx.Response(200, json = {"tokens": list(range(20))})

    client_type = httpx.Client
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx, "Client", lambda **kw: client_type(transport = transport, **kw))
    backend = _backend(20)
    backend.base_url = "http://llama.test"
    backend.is_loaded = True
    backend.markup_profile = None
    backend._auth_headers = None
    backend.count_chat_tokens = MethodType(LlamaCppBackend.count_chat_tokens, backend)
    payload = _payload()
    payload.messages = [{"role": "user", "content": "Hello"}]

    async def scenario():
        queue = LlamaAdmissionQueue("endpoint-failover")
        monkeypatch.setattr(inference, "get_llama_admission_queue", lambda _: queue)
        reservations = await asyncio.gather(
            *(
                inference._reserve_counted_gguf_chat(
                    request = None,
                    llama_backend = backend,
                    payload = payload,
                    messages = payload.messages,
                )
                for _ in range(3)
            )
        )
        leases = [reservation.lease_nowait() for reservation, _ in reservations]
        try:
            assert all(lease is not None for lease in leases)
            assert queue.snapshot().committed == 3 * (20 + 1024)
        finally:
            for (reservation, _), lease in zip(reservations, leases):
                if lease is not None:
                    lease.release()
                else:
                    reservation.cancel()

    asyncio.run(scenario())
    assert calls.count("/v1/chat/completions/input_tokens") == 3
    assert calls.count("/apply-template") == (0 if failed_path == "legacy" else 3)


def test_native_count_preserves_rendering_options_and_tools(monkeypatch):
    import httpx
    import json
    from types import MethodType
    from core.inference.llama_cpp import LlamaCppBackend

    bodies = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions/input_tokens"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json = {"input_tokens": 22560})

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client_type(transport = httpx.MockTransport(respond), **kw)
    )
    backend = _backend(22560)
    backend.base_url = "http://llama.test"
    backend.is_loaded = True
    backend.markup_profile = None
    backend._auth_headers = None
    backend.count_chat_tokens = MethodType(LlamaCppBackend.count_chat_tokens, backend)
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Partial"}]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    assert (
        backend.count_chat_tokens(
            messages,
            system = "System instructions",
            tools = tools,
            strict = True,
            chat_template_kwargs = {"enable_thinking": False},
            continue_final_message = True,
            prefer_native = True,
        )
        == 22560
    )
    assert bodies == [
        {
            "messages": [{"role": "system", "content": "System instructions"}] + messages,
            "tools": tools,
            "chat_template_kwargs": {"enable_thinking": False},
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
    ]


def test_all_count_endpoints_unavailable_still_reserves_pool(monkeypatch):
    import httpx
    from types import MethodType
    from core.inference.llama_cpp import LlamaCppBackend

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: client_type(
            transport = httpx.MockTransport(lambda _: httpx.Response(503)), **kw
        ),
    )
    backend = _backend(0)
    backend.base_url = "http://llama.test"
    backend.is_loaded = True
    backend.markup_profile = None
    backend._auth_headers = None
    backend.count_chat_tokens = MethodType(LlamaCppBackend.count_chat_tokens, backend)
    payload = _payload()
    assert inference._count_gguf_admission_prompt(backend, payload, payload.messages) == 30000


def test_media_keeps_existing_allowance_without_native_embedding_count():
    backend = _backend(20)
    payload = _payload()
    payload.messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        }
    ]
    count = inference._count_gguf_admission_prompt(backend, payload, payload.messages)
    assert count == 20 + inference._openai_llama_admission_image_tokens(backend)
    assert backend.count_chat_tokens.call_args.kwargs["prefer_native"] is False


def test_counting_recovers_after_a_complete_outage(monkeypatch):
    async def scenario():
        queue = LlamaAdmissionQueue("count-recovery")
        monkeypatch.setattr(inference, "get_llama_admission_queue", lambda _: queue)
        backend = _backend(20)
        payload = _payload()
        payload.messages = [{"role": "user", "content": "Hello"}]
        backend.count_chat_tokens.side_effect = RuntimeError("both count paths unavailable")
        first, _ = await inference._reserve_counted_gguf_chat(
            request = None, llama_backend = backend, payload = payload, messages = payload.messages
        )
        lease = first.lease_nowait()
        assert lease is not None
        assert queue.snapshot().committed == 30000
        lease.release()
        backend.count_chat_tokens.side_effect = None
        reservations = [
            (
                await inference._reserve_counted_gguf_chat(
                    request = None, llama_backend = backend, payload = payload, messages = payload.messages
                )
            )[0]
            for _ in range(3)
        ]
        leases = [reservation.lease_nowait() for reservation in reservations]
        try:
            assert all(lease is not None for lease in leases)
            assert queue.snapshot().committed == 3 * (20 + 1024)
        finally:
            for reservation, lease in zip(reservations, leases):
                if lease is not None:
                    lease.release()
                else:
                    reservation.cancel()

    asyncio.run(scenario())


def test_exact_count_preserves_video_allowance(monkeypatch):
    backend = _backend(20)
    payload = _payload()
    monkeypatch.setattr(inference, "_conversation_video_clips", lambda _: 2)
    media = Mock(return_value = 4096)
    monkeypatch.setattr(inference, "_openai_llama_admission_media_tokens", media)
    assert inference._count_gguf_admission_prompt(backend, payload, payload.messages) == 4116
    assert media.call_args.kwargs["message_video_clips"] == 2


def test_prepared_message_parsing_runs_off_the_event_loop(monkeypatch):
    import threading

    loop_thread = threading.get_ident()
    parse = inference._openai_llama_admission_messages_for_estimate
    seen = []

    def checked(*args, **kwargs):
        seen.append(threading.get_ident())
        assert threading.get_ident() != loop_thread
        return parse(*args, **kwargs)

    monkeypatch.setattr(inference, "_openai_llama_admission_messages_for_estimate", checked)

    async def scenario():
        payload = _payload()
        reservation, _ = await inference._reserve_counted_gguf_chat(
            request = None, llama_backend = _backend(20), payload = payload, messages = payload.messages
        )
        reservation.cancel()

    asyncio.run(scenario())
    assert seen and all(thread != loop_thread for thread in seen)


@pytest.mark.parametrize("continue_request", [False, True])
def test_tool_recost_rechecks_continuation_after_tool_result(monkeypatch, continue_request):
    async def scenario():
        queue = LlamaAdmissionQueue("continuation-recost")
        monkeypatch.setattr(inference, "get_llama_admission_queue", lambda _: queue)
        backend = _backend(12000)
        payload = _payload()
        payload.continue_final_message = continue_request
        payload.messages = [
            {"role": "user", "content": "Look up the result"},
            {"role": "assistant", "content": "Let me check."},
        ]
        reservation, _ = await inference._reserve_counted_gguf_chat(
            request = None,
            llama_backend = backend,
            payload = payload,
            messages = payload.messages,
            tool_loop = True,
        )
        lease = reservation.lease_nowait()
        assert lease is not None
        try:
            assert (
                backend.count_chat_tokens.call_args.kwargs["continue_final_message"]
                is continue_request
            )
            call = {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
            conversation = [
                payload.messages[0],
                {**payload.messages[1], "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Found it"},
            ]
            inference._openai_llama_admission_recost(
                reservation,
                conversation,
                request = None,
                llama_backend = backend,
                payload = payload,
                output_tokens = 1024,
                count_prepared_prompt = True,
            )
            assert backend.count_chat_tokens.call_count == 2
            assert backend.count_chat_tokens.call_args.kwargs["continue_final_message"] is False
            assert queue.snapshot().committed == 13024
        finally:
            lease.release()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "disabled", ["UNSLOTH_LLAMA_ADMISSION_CONTROL", "UNSLOTH_LLAMA_ADMISSION_KV_BUDGET"]
)
def test_disabled_admission_skips_tool_round_count(monkeypatch, disabled):
    monkeypatch.setenv(disabled, "0")

    async def scenario():
        backend = _backend(20)
        payload = _payload()
        reservation, _ = await inference._reserve_counted_gguf_chat(
            request = None,
            llama_backend = backend,
            payload = payload,
            messages = payload.messages,
        )
        lease = reservation.lease_nowait()
        assert lease is not None
        try:
            inference._openai_llama_admission_recost(
                reservation,
                payload.messages,
                request = None,
                llama_backend = backend,
                payload = payload,
                count_prepared_prompt = True,
            )
            backend.count_chat_tokens.assert_not_called()
        finally:
            lease.release()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_during_count", [False, True])
def test_cancelled_tool_round_does_not_count_or_recost(cancel_during_count):
    import threading

    cancel = threading.Event()
    backend = _backend(20)
    payload = _payload()
    lease = Mock()
    reservation = SimpleNamespace(lease_nowait = lambda: lease)
    if cancel_during_count:

        def count(*args, **kwargs):
            assert kwargs["should_abort"]() is False
            cancel.set()
            assert kwargs["should_abort"]() is True
            return 20

        backend.count_chat_tokens.side_effect = count
    else:
        cancel.set()
    inference._openai_llama_admission_recost(
        reservation,
        payload.messages,
        request = None,
        llama_backend = backend,
        payload = payload,
        cancel_event = cancel,
        count_prepared_prompt = True,
    )
    assert backend.count_chat_tokens.call_count == int(cancel_during_count)
    lease.recost_waiting.assert_not_called()


def test_stop_after_native_count_failure_skips_fallback(monkeypatch):
    import threading
    import httpx
    from types import MethodType
    from core.inference.llama_cpp import LlamaCppBackend

    cancel = threading.Event()
    calls = []

    def respond(request):
        calls.append(request.url.path)
        cancel.set()
        return httpx.Response(503)

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client_type(transport = httpx.MockTransport(respond), **kw)
    )
    backend = _backend(20)
    backend.base_url = "http://llama.test"
    backend.is_loaded = True
    backend.markup_profile = None
    backend._auth_headers = None
    backend.count_chat_tokens = MethodType(LlamaCppBackend.count_chat_tokens, backend)
    payload = _payload()
    lease = Mock()
    inference._openai_llama_admission_recost(
        SimpleNamespace(lease_nowait = lambda: lease),
        payload.messages,
        request = None,
        llama_backend = backend,
        payload = payload,
        cancel_event = cancel,
        count_prepared_prompt = True,
    )
    assert calls == ["/v1/chat/completions/input_tokens"]
    lease.recost_waiting.assert_not_called()


def test_cancel_during_stream_admission_exits_tracker(monkeypatch):
    import threading
    from starlette.requests import Request
    from .llama_backend_double import FakeLlamaCppBackend
    from models.inference import ChatCompletionRequest
    from state import active_generations

    entered = threading.Event()
    release = threading.Event()

    class Backend(FakeLlamaCppBackend):
        context_length = 30000
        base_url = "http://cancel-count-test"
        effective_parallel_slots = 3

        def _request_reasoning_kwargs(self, *args):
            return {}

        def count_chat_tokens(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return 20

    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: Backend())
    monkeypatch.setattr(inference, "_effective_enable_tools", lambda *a, **kw: False)
    monkeypatch.setattr(inference, "_CANCEL_REGISTRY", {})
    monkeypatch.setattr(active_generations, "_ACTIVE", {})
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "Hello"}],
        stream = True,
        enable_tools = False,
        cancel_id = "count-cancel",
        thread_id = "count-cancel",
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [],
            "query_string": b"",
            "state": {},
        }
    )

    async def scenario():
        task = asyncio.create_task(
            inference.produce_openai_chat_completions(
                payload,
                request,
                "test-user",
                cancel_on_disconnect = True,
            )
        )
        try:

            async def wait_for_count():
                while not entered.is_set():
                    if task.done():
                        await task
                        pytest.fail("Count was never entered")
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(wait_for_count(), timeout = 2)
            assert active_generations.count() == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not inference._CANCEL_REGISTRY
            assert active_generations.count() == 0
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

    asyncio.run(scenario())
