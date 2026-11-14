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


def test_a_long_interrupted_temporary_reply_is_estimated_in_full():
    from core.inference.api_monitor import TemporaryUsage, temporary_usage_scope

    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    with temporary_usage_scope(TemporaryUsage("thread", "external", "message")):
        entry = monitor.start(
            endpoint = "/v1/chat/completions", method = "POST", model = "m", prompt = "", subject = "o"
        )
    for _ in range(4):
        monitor.append_reply(entry, "r" * 12_000)
    monitor.finish(entry, "cancelled")
    [receipt] = receipts
    assert receipt.completion_tokens == 12_000


def test_closing_a_temporary_stream_closes_the_stream_it_wraps():
    import asyncio

    from core.inference.api_monitor import TemporaryUsage

    events = []

    async def inner():
        try:
            yield b"one"
            yield b"two"
        finally:
            events.append("inner closed")

    async def run():
        outer = inference_route._scoped_stream(inner(), TemporaryUsage("t", "local"))
        assert await outer.__anext__() == b"one"
        await outer.aclose()
        events.append("outer closed")

    asyncio.run(run())
    assert events == ["inner closed", "outer closed"]


def test_a_cancelled_api_key_stream_settles_an_estimate():
    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    entry = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "p" * 40,
        subject = "owner",
        via_api_key = True,
    )
    monitor.append_reply(entry, "r" * 40)
    monitor.finish(entry, "cancelled")
    [receipt] = receipts
    assert (receipt.prompt_tokens, receipt.completion_tokens) == (10, 10)


def test_reasoning_counts_toward_an_interrupted_estimate():
    from core.inference.api_monitor import TemporaryUsage, temporary_usage_scope

    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    with temporary_usage_scope(TemporaryUsage("thread", "local", "message")):
        entry = monitor.start(
            endpoint = "/v1/chat/completions", method = "POST", model = "m", prompt = "", subject = "o"
        )
    monitor_before = inference_route.api_monitor
    inference_route.api_monitor = monitor
    try:
        chunk = {"choices": [{"index": 0, "delta": {"reasoning_content": "t" * 400}}]}
        inference_route._monitor_openai_chunk(entry, chunk, None)
    finally:
        inference_route.api_monitor = monitor_before
    monitor.finish(entry, "cancelled")
    [receipt] = receipts
    assert receipt.completion_tokens == 100
    assert monitor.get(entry)["reply"] == ""


def test_reported_input_keeps_its_count_while_missing_output_is_estimated():
    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    entry = monitor.start(
        endpoint = "/v1/messages",
        method = "POST",
        model = "m",
        prompt = "p" * 4000,
        subject = "owner",
        via_api_key = True,
    )
    monitor.set_usage(entry, prompt_tokens = 10)
    monitor.append_reply(entry, "r" * 400)
    monitor.finish(entry, "cancelled")
    [receipt] = receipts
    assert (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens) == (
        10,
        100,
        110,
    )


def test_output_after_the_last_usage_report_is_still_charged():
    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    entry = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "",
        subject = "o",
        via_api_key = True,
    )
    monitor.append_reply(entry, "a" * 40)
    monitor.set_usage(entry, prompt_tokens = 90, completion_tokens = 10, total_tokens = 100)
    monitor.append_reply(entry, "b" * 400)
    monitor.finish(entry, "cancelled")
    [receipt] = receipts
    assert (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens) == (
        90,
        110,
        200,
    )


def test_a_finished_stream_adds_nothing_to_its_reported_usage():
    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    entry = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "",
        subject = "o",
        via_api_key = True,
    )
    monitor.append_reply(entry, "a" * 400)
    monitor.set_usage(entry, prompt_tokens = 90, completion_tokens = 7, total_tokens = 97)
    monitor.finish(entry)
    [receipt] = receipts
    assert (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens) == (90, 7, 97)


def test_anthropic_thinking_counts_toward_an_interrupted_estimate():
    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    entry = monitor.start(
        endpoint = "/v1/messages",
        method = "POST",
        model = "m",
        prompt = "",
        subject = "o",
        via_api_key = True,
    )
    monitor.set_usage(entry, prompt_tokens = 10)
    monitor_before = inference_route.api_monitor
    inference_route.api_monitor = monitor
    try:
        inference_route._monitor_anthropic_payload(
            entry,
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "t" * 400},
            },
        )
    finally:
        inference_route.api_monitor = monitor_before
    monitor.finish(entry, "cancelled")
    [receipt] = receipts
    assert (receipt.prompt_tokens, receipt.completion_tokens) == (10, 100)


def test_cancelling_a_temporary_stream_cancels_the_stream_it_wraps():
    import asyncio

    from core.inference.api_monitor import TemporaryUsage

    events = []

    async def inner():
        try:
            yield b"one"
            yield b"two"
        except asyncio.CancelledError:
            events.append("inner cancelled")
            raise
        except GeneratorExit:
            events.append("inner closed without cancellation")
            raise

    async def run():
        outer = inference_route._scoped_stream(inner(), TemporaryUsage("t", "local"))
        assert await outer.__anext__() == b"one"
        try:
            await outer.athrow(asyncio.CancelledError())
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert events == ["inner cancelled"]


def test_every_gguf_reasoning_branch_counts_its_length():
    import inspect
    import re

    source = inspect.getsource(inference_route)
    branches = re.findall(
        r"if (reasoning_delta|final_reasoning|fr):\n(?:\s*#.*\n)*\s*api_monitor\.mark_first_token\(([^)]*)\)",
        source,
    )
    assert len(branches) >= 11
    assert all("output_chars" in args for _, args in branches), branches


def _api_key_entry(monitor):
    return monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "",
        subject = "o",
        via_api_key = True,
    )


def test_streamed_tool_arguments_count_in_full_and_once():
    for flush in (False, True):
        receipts = []
        monitor = ApiMonitor()
        monitor.set_terminal_callback(receipts.append)
        entry = _api_key_entry(monitor)
        monitor.accumulate_openai_tool_call(
            entry,
            choice_index = 0,
            tool_index = 0,
            call_id = "c",
            name_fragment = None,
            arguments_fragment = "a" * 4000,
        )
        if flush:
            monitor_before = inference_route.api_monitor
            inference_route.api_monitor = monitor
            try:
                inference_route._flush_monitor_stream_tool_calls(entry)
            finally:
                inference_route.api_monitor = monitor_before
        monitor.finish(entry, "cancelled")
        [receipt] = receipts
        assert receipt.completion_tokens == 1000, flush


def test_streamed_tool_argument_frames_count_toward_a_stopped_estimate(monkeypatch):
    class _Backend(FakeLlamaCppBackend):
        supports_tools = True
        context_length = 8192
        base_url = "http://token-limit-tool-args"
        effective_parallel_slots = 2

        def count_chat_tokens(self, *args, **kwargs):
            return 20

        def _request_reasoning_kwargs(self, *args, **kwargs):
            return {}

        def generate_chat_completion_with_tools(self, **kwargs):
            yield {"type": "tool_args", "tool_name": "render_html", "text": "h" * 300}
            yield {"type": "tool_args", "tool_name": "render_html", "text": "h" * 100}
            yield {"type": "tool_output", "tool_name": "render_html", "text": "o" * 9000}

    monitor = ApiMonitor()
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "render_html"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    response = TestClient(app).post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "draw"}],
            "stream": True,
            "enable_tools": True,
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    [entry] = monitor.snapshot()
    # The 400 generated argument characters count; the tool's own output does not.
    assert monitor._find_locked(entry["id"]).reply_chars == 400
