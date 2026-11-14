# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference.api_monitor import ApiMonitor, override_claim_scope
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


def _usage(
    prompt = None,
    completion = None,
    total = None,
    **extra,
):
    return (
        "usage",
        {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total, **extra},
    )


@pytest.mark.parametrize(
    # Session (UI) and API-key requests are both charged by their receipt.
    "session, prompt, steps, status, expected",
    [
        # Stopped before any usage report: estimate both sides from text length.
        (True, 400, [("reply", 80)], "cancelled", (100, 20, 120)),
        (False, 40, [("reply", 40)], "cancelled", (10, 10, 20)),
        # The display buffer is capped; the estimate is not.
        (True, 0, [("reply", 12_000)] * 4, "cancelled", (0, 12_000, 12_000)),
        # Generated text the reply buffer never shows still counts.
        (True, 0, [("reasoning", 400)], "cancelled", (0, 100, 100)),
        (True, 0, [("hidden", 400)], "cancelled", (0, 100, 100)),
        (False, 0, [("message_reasoning", 400)], "completed", (0, 100, 100)),
        # Calls past the preview's call cap are still generated output.
        (False, 0, [("tool_call", (i, 40)) for i in range(70)], "cancelled", (0, 700, 700)),
        # Whole-response calls: the preview trims arguments to 500 characters, the estimate does not.
        (False, 0, [("message_tool_call", 3986)], "completed", (0, 1000, 1000)),
        # A call beside visible text: 4 text characters plus the whole call.
        (False, 0, [("message_text_and_tool_call", 3986)], "completed", (0, 1001, 1001)),
        (False, 0, [("anthropic_tool_use", 3986)], "completed", (0, 1000, 1000)),
        # A stop during thinking returns only thinking blocks: they are generated output.
        (False, 0, [("anthropic_thinking", 400)], "cancelled", (0, 100, 100)),
        # A tool loop's earlier passes reported after a later one was interrupted mid-stream.
        (
            True,
            0,
            [("reply", 40), ("reply", 400), _usage(90, 10, 100, reported_chars = 40)],
            "cancelled",
            (90, 110, 200),
        ),
        # An adapter's all-zero placeholder (no stats yet) is not a report.
        (
            False,
            0,
            [
                ("reply", 400),
                ("route_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            ],
            "cancelled",
            (0, 100, 100),
        ),
        (False, 0, [_usage(10), ("thinking", 400)], "cancelled", (10, 100, 110)),
        (False, 0, [("tool_args", 4000)], "cancelled", (0, 1000, 1000)),
        (False, 0, [("tool_args", 4000), ("flush", 0)], "cancelled", (0, 1000, 1000)),
        # Anthropic reports input at message_start: keep it, estimate the missing output.
        (False, 4000, [_usage(10), ("reply", 400)], "cancelled", (10, 100, 110)),
        # Only output after the last report is added; a finished stream's report stands.
        (
            False,
            0,
            [("reply", 40), _usage(90, 10, 100), ("reply", 400)],
            "cancelled",
            (90, 110, 200),
        ),
        (False, 0, [("reply", 400), _usage(90, 7, 97)], "completed", (90, 7, 97)),
        # A reported zero is real (max_tokens 0 scoring), not missing usage.
        (
            False,
            4,
            [
                ("reply", 17),
                ("route_usage", {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}),
            ],
            "completed",
            (1, 0, 1),
        ),
    ],
)
def test_a_receipt_settles_reported_and_estimated_usage(
    monkeypatch, session, prompt, steps, status, expected
):
    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    entry = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "m",
        prompt = "p" * prompt,
        subject = "owner",
        via_api_key = not session,
    )
    for kind, value in steps:
        if kind == "reply":
            monitor.append_reply(entry, "r" * value)
        elif kind == "usage":
            monitor.set_usage(entry, **value)
        elif kind == "route_usage":
            inference_route._monitor_usage(entry, value)
        elif kind == "hidden":
            monitor.count_output(entry, value)
        elif kind == "message_reasoning":
            message = {"role": "assistant", "content": "", "reasoning_content": "t" * value}
            inference_route._monitor_openai_chunk(
                entry, {"choices": [{"index": 0, "message": message}]}
            )
        elif kind in ("message_tool_call", "message_text_and_tool_call"):
            call = {"id": "c", "function": {"name": "f", "arguments": "a" * value}}
            content = "hell" if kind == "message_text_and_tool_call" else None
            message = {"role": "assistant", "content": content, "tool_calls": [call]}
            inference_route._monitor_openai_chunk(
                entry, {"choices": [{"index": 0, "message": message}]}
            )
        elif kind in ("anthropic_tool_use", "anthropic_thinking"):
            if kind == "anthropic_thinking":
                block = {"type": "thinking", "thinking": "t" * value, "signature": ""}
            else:
                block = {"type": "tool_use", "id": "t", "name": "f", "input": "a" * value}
            response = SimpleNamespace(body = json.dumps({"content": [block]}).encode())
            inference_route._monitor_anthropic_json_response(response, entry)
        elif kind == "tool_call":
            index, chars = value
            monitor.accumulate_openai_tool_call(
                entry,
                choice_index = 0,
                tool_index = index,
                call_id = f"c{index}",
                name_fragment = None,
                arguments_fragment = "a" * chars,
            )
        elif kind == "reasoning":
            chunk = {"choices": [{"index": 0, "delta": {"reasoning_content": "t" * value}}]}
            inference_route._monitor_openai_chunk(entry, chunk, None)
        elif kind == "thinking":
            delta = {"type": "thinking_delta", "thinking": "t" * value}
            inference_route._monitor_anthropic_payload(
                entry, {"type": "content_block_delta", "delta": delta}
            )
        elif kind == "tool_args":
            monitor.accumulate_openai_tool_call(
                entry,
                choice_index = 0,
                tool_index = 0,
                call_id = "c",
                name_fragment = None,
                arguments_fragment = "a" * value,
            )
        else:
            inference_route._flush_monitor_stream_tool_calls(entry)
    monitor.finish(entry, status)
    [receipt] = receipts
    assert (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens) == expected


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


@pytest.mark.parametrize("healed", [False, True])
def test_responses_tool_arguments_count_once(monkeypatch, healed):
    from types import SimpleNamespace

    from models.inference import ChatCompletionRequest, ResponsesRequest

    monitor = ApiMonitor()
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    entry = monitor.start(
        endpoint = "/v1/responses", method = "POST", model = "m", prompt = "", via_api_key = True
    )
    translator = inference_route._ResponsesStreamTranslator(
        payload = ResponsesRequest(model = "m", input = "hi"),
        chat_req = ChatCompletionRequest(model = "m", messages = [{"role": "user", "content": "hi"}]),
        request = SimpleNamespace(state = SimpleNamespace()),
        monitor_id = entry,
        resp_id = "resp",
        created_at = 0,
        model = "m",
        context_length = None,
        parse_think_markers = False,
        healer = None,
    )
    call = {"id": "c1", "type": "function", "function": {"name": "edit", "arguments": "a" * 400}}
    if healed:
        translator._healed_event_sse([("call", call)])
    else:
        delta = {"tool_calls": [{"index": 0, **call}]}
        list(translator.feed({"choices": [{"index": 0, "delta": delta}]}))
    streamed = monitor._find_locked(entry).reply_chars
    list(translator.finish())
    assert streamed == monitor._find_locked(entry).reply_chars == 404


def test_safetensors_finalized_tool_calls_count_unless_their_arguments_streamed():
    import inspect

    source = inspect.getsource(inference_route)
    # An interrupted safetensors tool stream reports its earlier passes against their watermark.
    assert "reported_chars = _sf_reported_chars if cancel_event.is_set() else None" in source
    # Non-streaming reasoning never reaches the reply preview, so it is counted beside it.
    assert 'sum(len(reply.get("reasoning_content") or "") for reply in _monitor_replies)' in source
    assert '_sf_streamed_calls.add(event.get("tool_call_id"))' in source
    assert 'event.get("tool_call_id") not in _sf_streamed_calls' in source
    # A finalized call runs after its pass ended, so that pass's report already covers it.
    assert "Executed after its pass ended, so that pass's reported usage covers it." in source
    # The whole safetensors reply's thinking is split out of the preview, so it is counted beside it.
    assert "api_monitor.count_output(monitor_id, len(_reasoning_text))" in source
    # A stop mid-pass in the JSON drain keeps the kept turn outside the earlier passes' report.
    assert "reported_chars = None if _stats is not _sf_text_stats else 0," in source
    assert '_sf_text_stats = _sf_stats_holder.get("stats")' in source


def test_a_trimmed_prompt_still_estimates_its_full_length(monkeypatch):
    from core.inference import api_monitor as monitor_module

    monkeypatch.setattr(monitor_module, "_MAX_PROMPT_BYTES", 10_000)
    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)

    def start(prompt):
        return monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "m",
            prompt = prompt,
            subject = "owner",
            via_api_key = True,
        )

    entry = start("p" * 40_000)
    start("q" * 4_000)  # retention trims the older running prompt to a preview
    monitor.append_reply(entry, "r" * 40)
    monitor.finish(entry, "cancelled")
    [receipt] = receipts
    assert receipt.prompt_tokens == 10_000


@pytest.mark.parametrize("silent_cancel", [False, True])
def test_a_chatgpt_subscription_stream_settles_a_receipt(monkeypatch, silent_cancel):
    import asyncio
    import json
    from types import SimpleNamespace

    from core.inference import openai_codex_auth, openai_codex_client
    from models.inference import ChatCompletionRequest

    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)

    def chunk(delta, usage = None):
        body = {"id": "c", "object": "chat.completion.chunk", "model": "gpt-x"}
        body["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
        if usage:
            body["usage"] = usage
        return "data: " + json.dumps(body)

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self, **kwargs):
            if silent_cancel:
                # The client returns on cancel without yielding anything more.
                kwargs["cancel_event"].set()
                return
            yield chunk({"content": "hello"})
            yield chunk({}, {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37})

        async def close(self):
            pass

    async def resolve_access(*args, **kwargs):
        return "token", "acct"

    monkeypatch.setattr(openai_codex_client, "OpenAICodexClient", _Client)
    monkeypatch.setattr(
        openai_codex_client, "subscription_catalog_matches_account", lambda *a: True
    )
    monkeypatch.setattr(openai_codex_client, "restore_subscription_catalog", lambda *a: None)
    monkeypatch.setattr(openai_codex_client, "subscription_catalog_known", lambda *a: True)
    monkeypatch.setattr(openai_codex_client, "offered_subscription_model_ids", lambda *a: {"gpt-x"})
    monkeypatch.setattr(
        openai_codex_client, "offered_subscription_model", lambda *a: {"vision": False}
    )
    monkeypatch.setattr(openai_codex_auth, "load_oauth_bundle", lambda *a: {"account_id": "acct"})
    monkeypatch.setattr(openai_codex_auth, "resolve_access", resolve_access)
    monkeypatch.setattr(
        inference_route.providers_db,
        "get_provider",
        lambda pid: {
            "id": pid,
            "provider_type": "openai_codex",
            "base_url": openai_codex_auth.OPENAI_CODEX_API_BASE,
            "display_name": "ChatGPT",
            "is_enabled": True,
            "models": ["gpt-x"],
        },
    )
    monkeypatch.setattr(inference_route, "validate_provider_base_url", lambda url: url)
    monkeypatch.setattr(inference_route, "_request_has_api_key", lambda request: False)
    monkeypatch.setattr(inference_route, "_request_used_api_key", lambda request: False)
    monkeypatch.setattr(
        inference_route.account_access, "require_connection_model_allowed", lambda *a: None
    )

    class _Request:
        url = SimpleNamespace(path = "/v1/chat/completions")
        method = "POST"
        headers = {}
        state = SimpleNamespace()

        async def is_disconnected(self):
            return False

    payload = ChatCompletionRequest(
        model = "gpt-x",
        provider_id = "chatgpt",
        stream = True,
        messages = [{"role": "user", "content": "hi"}],
    )

    async def run():
        response = await inference_route._proxy_to_external_provider(payload, _Request(), "owner")
        async for _ in response.body_iterator:
            pass

    asyncio.run(run())
    [receipt] = receipts
    if silent_cancel:
        assert receipt.status == "cancelled"
        return
    assert (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens) == (30, 7, 37)
    assert receipt.via_api_key is False


def test_a_stopped_in_process_message_settles_its_partial_output(monkeypatch):
    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    backend = SimpleNamespace(models = {}, active_model_name = None)
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_orchestrator_public_model_id", lambda _backend: None)
    monkeypatch.setattr(inference_route, "_monitor_context_length", lambda: None)

    async def stopped_completion(_chat_req, request, _subject):
        # The user stops the run; the in-process backend still returns what it generated.
        request.state.generation_cancel_event.set()
        message = {"role": "assistant", "content": "r" * 400}
        return SimpleNamespace(body = json.dumps({"choices": [{"index": 0, "message": message}]}))

    monkeypatch.setattr(inference_route, "openai_chat_completions", stopped_completion)
    payload = inference_route.AnthropicMessagesRequest(
        model = "m", max_tokens = 16, messages = [{"role": "user", "content": "hi"}]
    )
    request = SimpleNamespace(
        state = SimpleNamespace(), url = SimpleNamespace(path = "/v1/messages"), method = "POST"
    )
    claim = {"spent": 1, "generated": False}
    with override_claim_scope(claim), pytest.raises(asyncio.CancelledError):
        asyncio.run(inference_route._anthropic_in_process_messages(payload, request, "owner"))
    [receipt] = receipts
    assert receipt.completion_tokens == 100
    assert claim["generated"]


def test_a_whole_responses_reply_counts_thinking_and_calls_beside_its_text(monkeypatch):
    from models.inference import ChatMessage, ResponsesRequest

    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    monkeypatch.setattr(inference_route, "_monitor_context_length", lambda: None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: None)
    monkeypatch.setattr(
        inference_route, "_responses_should_parse_think_markers", lambda *_args: False
    )
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": "a" * 3986}}
    message = {
        "role": "assistant",
        "content": "hell",
        "reasoning_content": "t" * 400,
        "tool_calls": [call],
    }

    async def completion(*_args):
        # No usage: the receipt is estimated from the generated text.
        return {"choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}]}

    monkeypatch.setattr(inference_route, "openai_chat_completions", completion)
    request = SimpleNamespace(
        state = SimpleNamespace(), url = SimpleNamespace(path = "/v1/responses"), method = "POST"
    )
    asyncio.run(
        inference_route._responses_non_streaming(
            ResponsesRequest(model = "m", input = "hi"),
            [ChatMessage(role = "user", content = "hi")],
            request,
            "owner",
        )
    )
    [receipt] = receipts
    # 4 visible + 400 thinking + 4000 for the call ("Tool call: f(...)").
    assert receipt.completion_tokens == 1101


def test_media_routes_are_admitted_against_the_allowance(monkeypatch):
    from fastapi import HTTPException

    from core.inference import token_limit

    def reached():
        raise HTTPException(status_code = 429, detail = "reached")

    monkeypatch.setattr(token_limit, "enforce", reached)
    request = SimpleNamespace(
        state = SimpleNamespace(), url = SimpleNamespace(path = "/v1/audio/transcriptions"), method = "POST"
    )

    async def transcribe():
        async with inference_route._monitored_media_request(
            request, model = "m", prompt = "", subject = "owner"
        ):
            raise AssertionError("ran past an exhausted allowance")

    with pytest.raises(HTTPException) as refused:
        asyncio.run(transcribe())
    assert refused.value.status_code == 429


def _gguf_tool_route(
    monkeypatch,
    events,
    *,
    parse_think = False,
    tools = True,
):
    class _Backend(FakeLlamaCppBackend):
        supports_tools = True
        context_length = 8192
        base_url = "http://token-limit-tool-drain"
        effective_parallel_slots = 2

        def count_chat_tokens(self, *args, **kwargs):
            return 20

        def _request_reasoning_kwargs(self, *args, **kwargs):
            return {}

        def generate_chat_completion_with_tools(self, **kwargs):
            for event in events:
                if event == "stop":
                    # The user stops the run while it waits on the approval.
                    for bucket in list(inference_route._CANCEL_REGISTRY.values()):
                        for cancel in list(bucket):
                            cancel.set()
                    event = {"type": "heartbeat"}
                yield event

        def generate_chat_completion(self, **kwargs):
            yield from events

    receipts = []
    monitor = ApiMonitor()
    monitor.set_terminal_callback(receipts.append)
    monkeypatch.setattr(inference_route, "api_monitor", monitor)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: tools)
    monkeypatch.setattr(
        inference_route, "_responses_should_parse_think_markers", lambda *_args: parse_think
    )

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
            "stream": False,
            "enable_tools": tools,
            "cancel_id": "token-limit-drain",
        },
    )
    assert response.status_code == 200, response.text
    [receipt] = receipts
    return receipt


def test_a_whole_tool_run_without_a_final_frame_keeps_each_pass_report(monkeypatch):
    receipt = _gguf_tool_route(
        monkeypatch,
        [
            {"type": "context_usage", "prompt_tokens": 900, "completion_tokens": 100},
            {
                "type": "tool_start",
                "tool_name": "web_search",
                "tool_call_id": "call_1",
                "arguments": {"query": "x"},
                "awaiting_confirmation": True,
            },
            {"type": "context_usage", "prompt_tokens": 1100, "completion_tokens": 200},
            # Stopped at the approval wait: no final metadata frame.
        ],
    )
    assert (receipt.prompt_tokens, receipt.completion_tokens) == (1100, 300)


@pytest.mark.parametrize("tools", [True, False])
def test_a_whole_reply_counts_thinking_split_from_it(monkeypatch, tools):
    thought = "<think>" + "t" * 400 + "</think>"
    receipt = _gguf_tool_route(
        monkeypatch,
        [{"type": "content", "text": thought} if tools else thought],
        parse_think = True,
        tools = tools,
    )
    assert receipt.completion_tokens == 100


def test_a_stopped_whole_tool_run_charges_its_reported_preamble_once(monkeypatch):
    receipt = _gguf_tool_route(
        monkeypatch,
        [
            {"type": "content", "text": "hi there"},
            {"type": "context_usage", "prompt_tokens": 900, "completion_tokens": 100},
            {
                "type": "tool_start",
                "tool_name": "web_search",
                "tool_call_id": "call_1",
                "arguments": {"query": "x"},
                "awaiting_confirmation": True,
            },
            "stop",
            {"type": "content", "text": "never sent"},
        ],
    )
    assert receipt.status == "cancelled"
    assert (receipt.prompt_tokens, receipt.completion_tokens) == (900, 100)
