# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import threading
from types import SimpleNamespace

import pytest
import core.inference.llama_cpp as llama_cpp
from core.inference.llama_admission import (
    LlamaAdmissionConfig,
    LlamaAdmissionQueue,
    reset_llama_admission_queues,
)


@pytest.fixture(autouse = True)
def reset():
    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


def reserve(queue, tokens):
    return queue.reserve(capacity = 4, config = LlamaAdmissionConfig(), tokens = tokens, budget = 35000)


def test_reclaimed_approval_admits_waiter_and_resume_waits_for_context():
    async def scenario():
        queue = LlamaAdmissionQueue("approval")
        first = reserve(queue, 29000).lease_nowait()
        second = reserve(queue, 8750)
        assert second.lease_nowait() is None
        assert first.park(cache_reclaimed = True)
        await asyncio.sleep(0)
        other = second.lease_nowait()
        assert other is not None
        assert queue.snapshot().committed == 8750
        resumed = asyncio.create_task(first.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        assert not resumed.done(), "spare slots do not justify exceeding the KV budget"
        other.release()
        await asyncio.wait_for(resumed, 1)
        assert queue.snapshot().committed == 29000
        first.release()
        assert queue.snapshot().committed == 0
        assert queue.is_idle()

    asyncio.run(scenario())


def test_failed_reclamation_keeps_context_reserved():
    async def scenario():
        queue = LlamaAdmissionQueue("retain")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        second = reserve(queue, 8750)
        assert second.lease_nowait() is None
        assert queue.snapshot().committed == 29000
        second.cancel()
        first.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_cancelled_resume_does_not_subtract_another_chats_tokens():
    async def scenario():
        queue = LlamaAdmissionQueue("cancel")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park(cache_reclaimed = True)
        other = reserve(queue, 8750).lease_nowait()
        cancel = threading.Event()
        cancel.set()
        await first.unpark_async(cancel_event = cancel)
        first.release()
        assert queue.snapshot().committed == 8750
        other.release()
        assert queue.is_idle()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"id_slot": 2, "n_erased": 123}, True),
        ({"id_slot": 1, "n_erased": 123}, False),
        ({"id_slot": 2}, False),
    ],
)
def test_engine_erasure_requires_matching_acknowledgement(monkeypatch, body, expected):
    backend = SimpleNamespace(base_url = "http://127.0.0.1:9999", _auth_headers = {})

    def post(url, **kwargs):
        assert url == backend.base_url + "/slots/2"
        assert kwargs["params"] == {"action": "erase"}
        assert kwargs["trust_env"] is False
        return SimpleNamespace(raise_for_status = lambda: None, json = lambda: body)

    monkeypatch.setattr(llama_cpp.httpx, "post", post)
    assert (
        llama_cpp.LlamaCppBackend.release_idle_chat_slot(backend, backend.base_url, 2) is expected
    )


def test_erasure_does_not_target_a_replacement_server(monkeypatch):
    backend = SimpleNamespace(base_url = "http://127.0.0.1:9999", _auth_headers = {})

    def post(*args, **kwargs):
        pytest.fail("must not erase the replacement server's slot")

    monkeypatch.setattr(llama_cpp.httpx, "post", post)
    assert not llama_cpp.LlamaCppBackend.release_idle_chat_slot(backend, "http://127.0.0.1:9998", 2)


def test_release_during_readmission_returns_newly_acquired_tokens():
    async def scenario():
        queue = LlamaAdmissionQueue("released-resume")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park(cache_reclaimed = True)
        other = reserve(queue, 8750).lease_nowait()
        resumed = asyncio.create_task(first.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        first.release()
        other.release()
        await asyncio.wait_for(resumed, 1)
        assert queue.snapshot().committed == 0
        assert queue.is_idle()

    asyncio.run(scenario())


def test_erasure_failure_keeps_reservation_conservative(monkeypatch):
    backend = SimpleNamespace(base_url = "http://127.0.0.1:9999", _auth_headers = {})

    def post(*args, **kwargs):
        raise TimeoutError("engine did not acknowledge")

    monkeypatch.setattr(llama_cpp.httpx, "post", post)
    assert not llama_cpp.LlamaCppBackend.release_idle_chat_slot(backend, backend.base_url, 2)
