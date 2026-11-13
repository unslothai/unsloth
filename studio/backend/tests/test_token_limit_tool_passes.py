# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference.api_monitor import ApiMonitor
import routes.inference as inference_route
from .llama_backend_double import FakeLlamaCppBackend


def test_a_tool_run_that_stops_before_its_final_frame_still_reports_every_pass(monkeypatch):
    class _Backend(FakeLlamaCppBackend):
        supports_tools = True
        context_length = 8192
        base_url = "http://token-limit-tool-passes"
        effective_parallel_slots = 2

        def count_chat_tokens(self, *args, **kwargs):
            return 20

        def _request_reasoning_kwargs(self, *args, **kwargs):
            return {}

        def generate_chat_completion_with_tools(self, **kwargs):
            yield {"type": "context_usage", "prompt_tokens": 900, "completion_tokens": 100}
            yield {
                "type": "tool_start",
                "tool_name": "web_search",
                "tool_call_id": "call_1",
                "arguments": {"query": "x"},
                "awaiting_confirmation": True,
            }
            yield {
                "type": "context_usage",
                "prompt_tokens": 1100,
                "completion_tokens": 200,
                "context_tokens": 1300,
            }
            # Stopped at the approval wait: no final metadata frame ever arrives.

    monitor = ApiMonitor()
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    response = TestClient(app).post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "search it"}],
            "stream": True,
            "enable_tools": True,
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    [entry] = monitor.snapshot()
    # The final frame would report the last prompt plus both passes' completions.
    assert (entry["prompt_tokens"], entry["completion_tokens"], entry["total_tokens"]) == (
        1100,
        300,
        1400,
    )


def test_a_shared_preview_reply_gets_a_stateless_receipt():
    from models.inference import ChatCompletionRequest
    from routes.preview import _sanitize_preview_payload

    payload = ChatCompletionRequest(
        model = "m",
        messages = [{"role": "user", "content": "hi"}],
        thread_id = "visitor-supplied",
        temporary_message_id = "visitor-supplied",
    )
    clean = _sanitize_preview_payload(payload, is_lora = False)
    assert clean.temporary_chat is True
    assert (clean.thread_id, clean.temporary_message_id, clean.temporary_parent_id) == (None,) * 3


def test_an_interrupted_temporary_reply_settles_an_estimate():
    from core.inference.api_monitor import TemporaryUsage, temporary_usage_scope

    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    with temporary_usage_scope(TemporaryUsage("thread", "external", "message")):
        entry = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "m",
            prompt = "p" * 400,
            subject = "owner",
        )
    monitor.append_reply(entry, "r" * 80)
    monitor.finish(entry, "cancelled")
    [receipt] = receipts
    assert (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens) == (
        100,
        20,
        120,
    )
