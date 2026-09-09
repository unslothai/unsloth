# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("core.agent_workspace.task_state")
from core.agent_workspace import task_runtime as runtime


@pytest.fixture
def provider(monkeypatch):
    from storage import providers_db, credential_secrets
    from core.inference import providers

    config = {
        "id": "provider",
        "provider_type": "custom",
        "is_enabled": True,
        "models": ["model"],
        "base_url": "http://127.0.0.1:12345/v1",
        "updated_at": 1,
    }
    credential = ["binding-one"]
    monkeypatch.setattr(providers_db, "get_provider", lambda _: dict(config))
    monkeypatch.setattr(credential_secrets, "get_provider_api_key_binding", lambda _: credential[0])
    monkeypatch.setattr(
        credential_secrets,
        "get_provider_api_key_with_binding",
        lambda _: ("test-secret", credential[0]),
    )
    monkeypatch.setattr(providers, "provider_model_runs_local_tools", lambda *_: True)
    return config, credential


def test_runtime_capture_contains_no_key_and_detects_provider_drift(provider):
    config, _ = provider
    snapshot = runtime.capture_runtime("provider", "model", "provider")
    assert "test-secret" not in str(snapshot) and "127.0.0.1" not in str(snapshot)
    config["base_url"] = "https://different.example/v1"
    with pytest.raises(runtime.TaskStateError, match = "configuration changed"):
        runtime.validate_runtime(snapshot)


def test_runtime_rejects_changed_credentials(provider):
    _, credential = provider
    snapshot = runtime.capture_runtime("provider", "model", "provider")
    credential[0] = "binding-two"
    with pytest.raises(runtime.TaskStateError, match = "credential changed"):
        runtime.validate_runtime(snapshot)


def test_subscription_runtime_without_request_token_caps_is_rejected(provider):
    config, _ = provider
    config["provider_type"] = "openai_codex"
    with pytest.raises(runtime.TaskStateError, match = "output token limit"):
        runtime.capture_runtime("provider", "model", "provider")


def test_same_backend_with_replaced_model_process_is_rejected(monkeypatch):
    backend = SimpleNamespace(
        base_url = "http://localhost:1234", model_identifier = "model", _process = object()
    )
    monkeypatch.setattr(runtime, "_llama", lambda _: backend)
    snapshot = runtime.capture_runtime("local", "model")
    backend._process = object()
    with pytest.raises(runtime.TaskStateError, match = "runtime changed"):
        runtime.validate_runtime(snapshot)


def _context(snapshot, budget = 2300):
    return SimpleNamespace(
        task = {"snapshot": {"runtime": snapshot}, "maxOutputTokens": budget},
        cancel_event = threading.Event(),
        check = lambda: None,
    )


def test_turn_caps_sum_to_attempt_budget_even_when_provider_omits_usage(provider, monkeypatch):
    from core.inference import external_provider

    caps = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def stream_chat_completion(self, **kwargs):
            caps.append(kwargs["max_tokens"])
            yield 'data: {"choices": [{"delta": {"content": "done"}}]}\n\n'

        async def close(self):
            pass

    monkeypatch.setattr(external_provider, "ExternalProviderClient", Client)
    context = _context(runtime.capture_runtime("provider", "model", "provider"))
    transport = runtime.TaskTransport(context)

    async def run():
        for _ in range(3):
            async for _line in transport.stream(
                messages = [], tools = [], tool_choice = None, cancel_event = context.cancel_event
            ):
                pass
        with pytest.raises(runtime.TaskStateError, match = "budget exhausted"):
            async for _line in transport.stream(
                messages = [], tools = [], tool_choice = None, cancel_event = context.cancel_event
            ):
                pass

    asyncio.run(run())
    assert caps == [1024, 1024, 252]
    assert transport.reserved == 2300 and transport.remaining == 0


def test_cancel_closes_stalled_provider_stream_and_client(provider, monkeypatch):
    from core.inference import external_provider

    started, closed = threading.Event(), threading.Event()
    clients_closed = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def stream_chat_completion(self, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        async def close(self):
            clients_closed.append(True)

    monkeypatch.setattr(external_provider, "ExternalProviderClient", Client)
    context = _context(runtime.capture_runtime("provider", "model", "provider"))

    async def run():
        async def cancel():
            while not started.is_set():
                await asyncio.sleep(0.01)
            context.cancel_event.set()

        cancel_task = asyncio.create_task(cancel())
        async for _ in runtime.TaskTransport(context).stream(
            messages = [], tools = [], tool_choice = None, cancel_event = context.cancel_event
        ):
            pass
        await cancel_task

    asyncio.run(asyncio.wait_for(run(), 2))
    assert closed.is_set() and clients_closed == [True]


def test_encrypted_credential_binding_rotates_even_when_plaintext_is_reused(monkeypatch):
    from storage import credential_secrets

    monkeypatch.setattr(
        credential_secrets, "get_or_create_credential_encryption_key", lambda: b"k" * 32
    )
    absent = credential_secrets.get_provider_api_key_binding("saved")
    credential_secrets.save_provider_api_key("saved", "test-only-value")
    value, first = credential_secrets.get_provider_api_key_with_binding("saved")
    assert value == "test-only-value"
    assert first == credential_secrets.get_provider_api_key_binding("saved") and first != absent
    credential_secrets.save_provider_api_key("saved", "test-only-value")
    value, second = credential_secrets.get_provider_api_key_with_binding("saved")
    assert value == "test-only-value" and second != first
    credential_secrets.delete_provider_api_key("saved")
    assert credential_secrets.get_provider_api_key_with_binding("saved") == (None, absent)


def test_local_task_reservation_retains_aggregate_kv_budget_and_prices_tools(monkeypatch):
    from core.inference import llama_admission

    llama_admission.reset_llama_admission_queues()
    queue = llama_admission.get_llama_admission_queue("kv-test")
    seen = []

    def reserve(**kwargs):
        seen.append(kwargs)
        return queue.reserve(**kwargs)

    monkeypatch.setattr(
        llama_admission, "get_llama_admission_queue", lambda _: SimpleNamespace(reserve = reserve)
    )
    backend = SimpleNamespace(
        base_url = "kv-test",
        effective_parallel_slots = 2,
        context_length = 2048,
        _kv_cache_context_total = 4096,
    )
    context = _context({})

    async def run():
        async with runtime._admission(
            backend,
            context,
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"function": {"description": "a" * 400}}],
            max_tokens = 100,
        ):
            pass

    asyncio.run(run())
    assert seen[0]["budget"] == 4096 and seen[0]["capacity"] == 2
    assert seen[0]["tokens"] > 200


def test_local_runtime_must_clear_idle_slots_before_delegation(monkeypatch):
    import sys

    backend = SimpleNamespace(
        is_loaded = True, model_identifier = "model", idle_slot_clearing_active = False
    )
    monkeypatch.setitem(
        sys.modules, "routes.inference", SimpleNamespace(get_llama_cpp_backend = lambda: backend)
    )
    with pytest.raises(runtime.TaskStateError, match = "idle model slots"):
        runtime.capture_runtime("local", "model")
