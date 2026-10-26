# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A generation limit must not turn an unfinished rewrite into an executable tool call."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from core.inference import tools
from core.inference.safetensors_agentic import run_safetensors_tool_loop
from core.inference.tool_loop_controller import TRUNCATED_TOOL_TURN_MESSAGE
from .test_llama_cpp_tool_loop import (
    _backend_and_payloads,
    _done,
    _finish,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
    _tool_call_sse,
)


_COMMAND = "cat > app.py <<'EOF'\n" + "".join(f"x_{i} = {i}\n" for i in range(44))
_PARTIAL = "<tool_call><function=terminal><parameter=command>" + _COMMAND
_CLOSED = _PARTIAL + "EOF\n</parameter></function></tool_call>"
_ORIGINAL = "".join(f"x_{i} = {i}\n" for i in range(175))


@pytest.mark.parametrize("reason", ["length", "content_filter"])
@pytest.mark.parametrize("shape", ["xml", "prefaced_xml", "structured", "siblings"])
def test_gguf_never_executes_calls_from_a_truncated_turn(monkeypatch, reason, shape):
    if shape == "structured":
        stream = [_tool_call_sse("terminal", {"command": _COMMAND}, "call_write")]
    elif shape == "siblings":
        stream = [_sse({"content": _CLOSED + _PARTIAL})]
    elif shape == "prefaced_xml":
        stream = [_sse({"content": "Updating your file. "}), _sse({"content": _PARTIAL})]
    else:
        stream = [_sse({"content": _PARTIAL})]
    stream += [_finish(reason), _done()]
    backend, payloads = _backend_and_payloads(
        monkeypatch, [stream, [_sse({"content": "Done."}), _finish("stop"), _done()]]
    )
    calls = _record_tool_calls(monkeypatch, "must not run")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Update app.py"}],
        [tools.TERMINAL_TOOL],
        permission_mode = "off",
        sandbox_level = "low",
    )
    assert calls == []
    assert len(payloads) == 1
    assert events[-1]["finish_reason"] == reason
    text = [event["text"] for event in events if event["type"] == "content"][-1]
    assert "These tool calls were not run" in text
    if shape == "prefaced_xml":
        assert text == "\n\n" + TRUNCATED_TOOL_TURN_MESSAGE
    starts = {event["tool_call_id"] for event in events if event["type"] == "tool_start"}
    ends = {event["tool_call_id"] for event in events if event["type"] == "tool_end"}
    assert starts <= ends


@pytest.mark.parametrize("heal", [True, False])
@pytest.mark.parametrize("reason", ["stop", "tool_calls"])
def test_gguf_complete_calls_still_run(monkeypatch, heal, reason):
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": _CLOSED}), _finish(reason), _done()],
            [_sse({"content": "Done."}), _finish("stop"), _done()],
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *args, **kwargs: 100)
    calls = _record_tool_calls(monkeypatch, "written")
    _run_tool_loop(
        backend,
        [{"role": "user", "content": "Update app.py"}],
        [tools.TERMINAL_TOOL],
        auto_heal_tool_calls = heal,
        permission_mode = "off",
        sandbox_level = "low",
    )
    assert len(calls) == 1


def test_truncated_gguf_edit_preserves_the_real_file(monkeypatch, tmp_path):
    target = tmp_path / "app.py"
    target.write_text(_ORIGINAL)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(tmp_path))
    arguments = {
        "path": "app.py",
        "edits": [
            {"old_string": _ORIGINAL, "new_string": "".join(_ORIGINAL.splitlines(True)[:44])}
        ],
    }
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_tool_call_sse("edit_file", arguments, "call_write"), _finish("length"), _done()],
            [_sse({"content": "Done."}), _finish("stop"), _done()],
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *args, **kwargs: 100)
    _run_tool_loop(
        backend,
        [{"role": "user", "content": "Update app.py"}],
        [tools.EDIT_FILE_TOOL],
        permission_mode = "off",
        session_id = "truncated-edit",
    )
    assert target.read_text() == _ORIGINAL


@pytest.mark.parametrize(
    "stats",
    [{"truncated": True}, {"finish_reason": "length"}, {"finish_reason": "content_filter"}],
)
@pytest.mark.parametrize("name", ["terminal", "render_html"])
def test_safetensors_and_mlx_truncated_turns_do_not_execute(stats, name):
    holder = {}
    calls = []

    def single_turn(messages, **kwargs):
        key = "command" if name == "terminal" else "code"
        prefix = f"<tool_call><function={name}><parameter={key}>"
        yield prefix
        yield prefix + "print('partial')"
        holder["stats"] = stats

    def execute(name, arguments, **kwargs):
        calls.append((name, arguments))
        return "must not run"

    events = list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "Update the file"}],
            tools = [{"type": "function", "function": {"name": name}}],
            execute_tool = execute,
            generation_stats_holder = holder,
            permission_mode = "off",
            sandbox_level = "low",
        )
    )
    assert calls == []
    assert "These tool calls were not run" in events[-1]["text"]
    starts = {event["tool_call_id"] for event in events if event["type"] == "tool_start"}
    ends = {event["tool_call_id"] for event in events if event["type"] == "tool_end"}
    assert starts <= ends


@pytest.mark.parametrize("stats", [{"truncated": False}, {"finish_reason": "stop"}, {}])
def test_safetensors_complete_turn_still_executes(stats):
    holder = {}
    calls = []

    def single_turn(messages, **kwargs):
        yield _CLOSED if not calls else "Done."
        holder["stats"] = stats

    def execute(name, arguments, **kwargs):
        calls.append((name, arguments))
        return "written"

    list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "Update the file"}],
            tools = [tools.TERMINAL_TOOL],
            execute_tool = execute,
            generation_stats_holder = holder,
            permission_mode = "off",
            sandbox_level = "low",
        )
    )
    assert len(calls) == 1


@pytest.mark.parametrize("backend_kind", ["gguf", "safetensors"])
@pytest.mark.parametrize("prefix", ["", "Updating your file."])
@pytest.mark.parametrize("name", ["web_search", "terminal", "render_html"])
def test_truncated_notice_sse_does_not_repeat_streamed_text(
    monkeypatch, backend_kind, prefix, name
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    import routes.inference as inference_route
    from .llama_backend_double import FakeLlamaCppBackend

    selected_tools = [{"type": "function", "function": {"name": name}}]
    if name == "web_search":
        call = '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
    else:
        key = "command" if name == "terminal" else "code"
        call = f"<function={name}><parameter={key}>" + _COMMAND
    split = call.index(_COMMAND) if name != "web_search" else len(call) // 2
    calls = []

    def execute(tool_name, arguments, **kwargs):
        calls.append((tool_name, arguments))
        return "must not run"

    if backend_kind == "gguf":
        stream = [_sse({"content": prefix})] if prefix else []
        if name == "render_html":
            stream.append(_tool_call_sse(name, {"code": _COMMAND}, "call_preview"))
        else:
            stream += [_sse({"content": call[:split]}), _sse({"content": call[split:]})]
        stream += [_finish("length"), _done()]
        backend, _ = _backend_and_payloads(monkeypatch, [stream])
        monkeypatch.setattr(tools, "execute_tool", execute)
        events = _run_tool_loop(
            backend,
            [{"role": "user", "content": "Update the file"}],
            selected_tools,
            permission_mode = "off",
            sandbox_level = "low",
        )
    else:
        holder = {}

        def single_turn(messages, **kwargs):
            if prefix:
                yield prefix
            yield prefix + call[:split]
            yield prefix + call
            holder["stats"] = {"truncated": True}

        events = list(
            run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "Update the file"}],
                tools = selected_tools,
                execute_tool = execute,
                generation_stats_holder = holder,
                permission_mode = "off",
                sandbox_level = "low",
            )
        )
    assert calls == []
    starts = [event for event in events if event["type"] == "tool_start"]
    if name == "render_html":
        assert starts, "exercise the cursor reset from a provisional card"
    elif name == "web_search":
        assert not starts, "exercise rejection without a provisional card"

    class EventBackend(FakeLlamaCppBackend):
        supports_tools = True
        context_length = 8192
        active_model_name = "sf-model"
        models = {"sf-model": {"context_length": 8192}}

        def generate_chat_completion_with_tools(self, **kwargs):
            yield from events
            if kwargs.get("stats_holder") is not None:
                kwargs["stats_holder"]["stats"] = {"truncated": True}

    event_backend = EventBackend()
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def select_tools(payload, **kwargs):
        return selected_tools

    monkeypatch.setattr(inference_route, "_select_request_tools", select_tools)
    if backend_kind == "gguf":
        monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: event_backend)
    else:
        monkeypatch.setattr(
            inference_route,
            "get_llama_cpp_backend",
            lambda: SimpleNamespace(is_loaded = False, supports_tools = False),
        )
        monkeypatch.setattr(inference_route, "get_inference_backend", lambda: event_backend)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *args, **kwargs: {"supports_tools": True, "supports_reasoning": False},
        )
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    with TestClient(app) as client:
        response = client.post(
            "/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "Update the file"}],
                "stream": True,
                "enable_tools": True,
                "permission_mode": "off",
                "sandbox_level": "low",
            },
            headers = {"X-Unsloth-Events": "1"},
        )
    assert response.status_code == 200, response.text
    chunks = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    visible = "".join(
        choice.get("delta", {}).get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    expected = (
        prefix + "\n\n" + TRUNCATED_TOOL_TURN_MESSAGE if prefix else TRUNCATED_TOOL_TURN_MESSAGE
    )
    assert visible == expected
    assert [
        choice["finish_reason"]
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("finish_reason")
    ] == ["length"]


@pytest.mark.parametrize("stream", [False, True], ids = ["json", "sse"])
@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param("", id = "empty"),
        pytest.param("Updating your file.", id = "short"),
        pytest.param("x" * 270, id = "long"),
    ],
)
def test_anthropic_truncated_notice_survives_without_a_provisional_card(
    monkeypatch, stream, prefix
):
    from routes.inference import _anthropic_tool_non_streaming, _anthropic_tool_stream

    selected_tools = [{"type": "function", "function": {"name": "web_search"}}]
    frames = [_sse({"content": prefix})] if prefix else []
    frames += [
        _sse({"content": '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'}),
        "data: "
        + json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 9, "total_tokens": 24},
            }
        )
        + "\n",
        _done(),
    ]
    backend, _ = _backend_and_payloads(monkeypatch, [frames])
    calls = _record_tool_calls(monkeypatch, "must not run")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Search"}],
        selected_tools,
        permission_mode = "off",
        sandbox_level = "low",
    )
    assert calls == []
    assert not any(event["type"] == "tool_start" for event in events)

    async def disconnected():
        return False

    def run_gen():
        yield from events

    async def consume():
        request = SimpleNamespace(is_disconnected = disconnected, scope = {})
        if stream:
            response = await _anthropic_tool_stream(
                request,
                threading.Event(),
                run_gen,
                "msg_notice",
                "test-model",
                openai_tools = selected_tools,
            )
            lines = [line async for line in response.body_iterator]
            chunks = [
                json.loads(line[6:])
                for frame in lines
                for line in frame.splitlines()
                if line.startswith("data: {")
            ]
            visible = "".join(
                chunk["delta"]["text"]
                for chunk in chunks
                if chunk.get("delta", {}).get("type") == "text_delta"
            )
            final = next(chunk for chunk in chunks if chunk["type"] == "message_delta")
            assert final["delta"]["stop_reason"] == "max_tokens"
            assert final["usage"]["output_tokens"] == 9
            return visible
        response = await _anthropic_tool_non_streaming(
            request,
            run_gen,
            "msg_notice",
            "test-model",
            openai_tools = selected_tools,
            cancel_event = threading.Event(),
        )
        body = json.loads(response.body)
        assert body["stop_reason"] == "max_tokens"
        assert body["usage"]["input_tokens"] == 15
        assert body["usage"]["output_tokens"] == 9
        return "".join(block["text"] for block in body["content"] if block["type"] == "text")

    expected = (
        prefix + "\n\n" + TRUNCATED_TOOL_TURN_MESSAGE if prefix else TRUNCATED_TOOL_TURN_MESSAGE
    )
    assert asyncio.run(consume()) == expected
