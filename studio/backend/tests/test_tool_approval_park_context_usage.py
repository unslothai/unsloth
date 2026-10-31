# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF run waiting on a tool approval keeps its llama-server slot parked.

The tool loop sends a ``context_usage`` meter frame between the gated ``tool_start`` and the wait
for the user's decision. That frame is not the answer: the slot must stay parked until a real
answer (``status`` / ``tool_end``) arrives, or every open approval prompt holds a slot.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import llama_admission
from core.inference.llama_admission import LlamaAdmissionLease
import routes.inference as inference_route
from .llama_backend_double import FakeLlamaCppBackend


@pytest.fixture
def lease_log(monkeypatch):
    log = []
    real_park = LlamaAdmissionLease.park
    real_unpark_async = LlamaAdmissionLease.unpark_async
    real_unpark = LlamaAdmissionLease.unpark

    def park(self):
        ok = real_park(self)
        log.append("park" if ok else "park-refused")
        return ok

    async def unpark_async(self, *args, **kwargs):
        log.append("unpark")
        return await real_unpark_async(self, *args, **kwargs)

    def unpark(self):
        log.append("unpark")
        return real_unpark(self)

    # Park budget is sized from the host's executor; pin it so the test does not depend on CPU count.
    monkeypatch.setattr(llama_admission, "_max_parked", lambda capacity: 4)
    monkeypatch.setattr(LlamaAdmissionLease, "park", park)
    monkeypatch.setattr(LlamaAdmissionLease, "unpark_async", unpark_async)
    monkeypatch.setattr(LlamaAdmissionLease, "unpark", unpark)
    return log


def _client(monkeypatch, log):
    class _Backend(FakeLlamaCppBackend):
        supports_tools = True
        context_length = 8192
        base_url = "http://approval-park-context-usage"
        effective_parallel_slots = 2

        def count_chat_tokens(self, *args, **kwargs):
            return 20

        def _request_reasoning_kwargs(self, *args, **kwargs):
            return {}

        def generate_chat_completion_with_tools(self, **kwargs):
            yield {
                "type": "tool_start",
                "tool_name": "web_search",
                "tool_call_id": "call_1",
                "arguments": {"query": "x"},
                "awaiting_confirmation": True,
            }
            yield {
                "type": "context_usage",
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "context_length": 8192,
            }
            # The loop now blocks on the user's decision.
            log.append("waiting-for-decision")
            yield {"type": "status", "text": "Searching"}
            yield {
                "type": "tool_end",
                "tool_name": "web_search",
                "tool_call_id": "call_1",
                "result": "r",
            }
            yield {"type": "content", "text": "done"}
            yield {"type": "metadata", "finish_reason": "stop"}

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_context_meter_before_the_wait_does_not_take_the_slot_back(monkeypatch, lease_log):
    response = _client(monkeypatch, lease_log).post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "search it"}],
            "stream": True,
            "enable_tools": True,
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    assert "context_usage" in response.text, "the meter frame must still reach the UI"
    assert "park" in lease_log, lease_log
    wait = lease_log.index("waiting-for-decision")
    park = lease_log.index("park")
    assert park < wait, lease_log
    assert "unpark" not in lease_log[park:wait], (
        "the slot was taken back before the user answered: " + repr(lease_log)
    )
    # The answer (status / tool_end) takes the slot back.
    assert "unpark" in lease_log[wait:], lease_log
