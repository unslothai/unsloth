# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from core.inference.engine_adapters import (
    ADAPTERS,
    launch_arguments,
    reasoning_parser_for_template,
)
from core.inference.managed_engine import ManagedEngine, validate_model

_QWEN3 = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
    "{%- endif %}{%- endif %}"
)


@pytest.mark.parametrize(
    "template,engine,parser",
    [
        (_QWEN3, "vllm", "qwen3"),
        (_QWEN3, "sglang", None),
        ("<|channel|>analysis<|message|>", "vllm", "openai_gptoss"),
        ("<｜Assistant｜><think></think>", "vllm", "deepseek_r1"),
        # V3.1 gates on `thinking` and answers untagged when it is off.
        (
            "<｜Assistant｜>{% if thinking %}<think>{% else %}</think>{% endif %}",
            "vllm",
            "deepseek_v3",
        ),
        ("<｜Assistant｜>{{ reasoning_effort }}<think></think>", "vllm", None),
        # QwQ / Qwen3 Thinking-2507 open <think> in the prompt, which deepseek_r1 expects.
        ("<|im_start|>assistant\n<think>\n{# </think> #}", "vllm", "deepseek_r1"),
        ("<|im_start|>assistant", "vllm", None),
        (None, "vllm", None),
    ],
)
def test_reasoning_parser_follows_the_template(template, engine, parser):
    assert reasoning_parser_for_template(template, engine) == parser


def test_reasoning_parser_reaches_only_the_vllm_argv():
    vllm, sglang = ADAPTERS["vllm"], ADAPTERS["sglang"]
    argv = vllm.command("/py", "org/m", 1, "k", 4096, 0.8, options = {"reasoning_parser": "qwen3"})
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert "--reasoning-parser" not in launch_arguments("vllm", "/py", "org/m", 1, "k", 4096)
    assert "--reasoning-parser" not in sglang.command(
        "/py", "org/m", 1, "k", 4096, 0.8, options = {"reasoning_parser": "qwen3"}
    )


def _model_dir(tmp_path, template):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    if template is not None:
        (tmp_path / "chat_template.jinja").write_text(template)
    return SimpleNamespace(is_local = True, path = str(tmp_path), identifier = str(tmp_path))


def test_validate_records_the_parser_and_template_flags(tmp_path):
    options = validate_model(_model_dir(tmp_path, _QWEN3), engine = "vllm")
    assert options["reasoning_parser"] == "qwen3"
    assert options["reasoning_flags"]["supports_reasoning"] is True
    assert options["reasoning_flags"]["reasoning_style"] == "enable_thinking"


def test_a_template_without_thinking_launches_as_before(tmp_path):
    options = validate_model(_model_dir(tmp_path, "<|im_start|>{{ m }}"), engine = "vllm")
    assert "reasoning_parser" not in options and "reasoning_flags" not in options


def test_reasoning_fields():
    engine = ManagedEngine("vllm")
    assert engine.reasoning_fields(True, "high", True) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    engine.reasoning_parser = "qwen3"
    assert engine.reasoning_fields() == {
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False}
    }
    assert engine.reasoning_fields(True) == {
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True}
    }
    assert engine.reasoning_fields(None, "xhigh", False) == {
        "chat_template_kwargs": {"preserve_thinking": False},
        "reasoning_effort": "xhigh",
    }


def test_preserved_thinking_is_replayed_as_vllm_reads_it():
    engine = ManagedEngine("vllm")
    history = [{"role": "assistant", "content": "4", "reasoning_content": "2+2"}]
    assert engine.replay_reasoning(history, True) == history
    engine.reasoning_parser = "qwen3"
    assert engine.replay_reasoning(history, False) == history
    # Moved, so the context fit does not price the thought twice.
    assert engine.replay_reasoning(history, True) == [
        {"role": "assistant", "content": "4", "reasoning": "2+2"}
    ]


@pytest.fixture
def thinking_peer():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta in ({"reasoning": "two"}, {"reasoning": " plus two"}, {"content": "\n\n4"}):
                self.wfile.write(
                    f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    engine = ManagedEngine("vllm")
    engine.base_url = f"http://127.0.0.1:{server.server_port}"
    engine.model = "org/model"
    engine.reasoning_parser = "qwen3"
    yield engine, requests
    server.shutdown()
    server.server_close()
    thread.join(2)


def test_streamed_reasoning_becomes_think_markup(thinking_peer):
    engine, requests = thinking_peer
    out = "".join(
        engine.generate(
            messages = [{"role": "user", "content": "2+2?"}],
            enable_thinking = True,
            reasoning_effort = "low",
        )
    )
    assert out == "<think>two plus two</think>4"
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": True, "thinking": True}
    assert requests[0]["reasoning_effort"] == "low"


def test_reasoning_controls_are_refused_only_without_a_parser():
    from models.inference import ChatCompletionRequest
    from routes.inference import _managed_engine_unsupported_controls

    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        enable_thinking = True,
        reasoning_effort = "high",
    )
    assert _managed_engine_unsupported_controls(payload, {"engine": "vllm"})
    assert not _managed_engine_unsupported_controls(
        payload, {"engine": "vllm", "reasoning_flags": {"supports_reasoning": True}}
    )
    payload = payload.model_copy(update = {"continue_final_message": True})
    assert _managed_engine_unsupported_controls(
        payload, {"engine": "vllm", "reasoning_flags": {"supports_reasoning": True}}
    )


def test_load_flags_come_from_the_engine_template():
    from routes.inference import _detect_safetensors_features

    flags = {
        "supports_reasoning": True,
        "reasoning_style": "enable_thinking_effort",
        "reasoning_always_on": False,
        "reasoning_effort_levels": ["low", "medium", "xhigh"],
        "supports_preserve_thinking": True,
        "preserve_thinking_default": False,
        "supports_tools": True,
    }
    backend = SimpleNamespace(
        active_model_name = "m",
        models = {"m": {"engine": "vllm", "supports_tools": False, "reasoning_flags": flags}},
    )
    assert _detect_safetensors_features(backend, None) == {**flags, "supports_tools": False}
    backend.models["m"].pop("reasoning_flags")
    assert _detect_safetensors_features(backend, None)["supports_reasoning"] is False


def test_managed_engine_chat_is_fitted_to_the_engine_window():
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {"engine": "vllm", "context_length": 2048}}
    calls = []

    def count(
        messages,
        system_prompt = "",
        **kwargs,
    ):
        calls.append(kwargs)
        return sum(len(str(m.get("content", ""))) // 4 for m in messages), "m"

    orchestrator.count_chat_tokens = count
    turn = "word " * 400
    messages = []
    for index in range(10):
        messages += [
            {"role": "user", "content": f"q{index} {turn}"},
            {"role": "assistant", "content": f"a{index} {turn}"},
        ]
    messages.append({"role": "user", "content": "latest"})
    fitted = orchestrator.compact_chat_context(
        messages,
        system_prompt = "Be brief.",
        context_overflow = "truncate_oldest",
        max_tokens = 256,
        enable_thinking = True,
    )
    assert fitted["messages"][-1]["content"] == "latest"
    assert len(fitted["messages"]) < len(messages)
    assert any(event.get("type") == "context_truncated" for event in fitted["events"])
    assert calls[0]["enable_thinking"] is True
    unchanged = orchestrator.compact_chat_context(messages, context_overflow = None, max_tokens = 256)
    assert unchanged["messages"] == messages


def test_tool_stream_reasoning_reaches_studio_as_reasoning_content():
    from routes.managed_engine_chat import _with_reasoning_content

    line = 'data: {"choices":[{"delta":{"reasoning":"hmm"}}]}'
    assert json.loads(_with_reasoning_content(line)[5:])["choices"][0]["delta"] == {
        "reasoning_content": "hmm"
    }
    plain = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
    assert _with_reasoning_content(plain) == plain


class _Engine:
    engine = "vllm"
    context = 2048
    reasoning_parser = "qwen3"
    replay_reasoning = ManagedEngine.replay_reasoning


def _tool_payload(**fields):
    from models.inference import ChatCompletionRequest
    return ChatCompletionRequest(
        model = "m",
        messages = [{"role": "user", "content": "hi"}],
        max_tokens = 256,
        **fields,
    )


def test_tool_loop_keeps_and_replays_reasoning_when_preserving():
    import asyncio

    from routes.managed_engine_chat import ManagedToolTransport

    sent = []

    class _Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'

        async def aclose(self):
            return None

    class _Stream:
        async def __aenter__(self):
            return _Response()

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def stream(
            self,
            method,
            url,
            headers = None,
            json = None,
        ):
            sent.append(json)
            return _Stream()

    engine = _Engine()
    engine.base_url, engine.headers = "http://engine", {}
    turn = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "because"},
    ]

    async def run(transport):
        return [
            line
            async for line in transport.stream(
                messages = turn, tools = None, tool_choice = None, cancel_event = asyncio.Event()
            )
        ]

    keep = ManagedToolTransport(_Client(), engine, {"model": "m"}, preserve_thinking = True)
    assert keep.preserves_reasoning is True
    asyncio.run(run(keep))
    assert sent[-1]["messages"][1]["reasoning"] == "because"
    drop = ManagedToolTransport(_Client(), engine, {"model": "m"}, preserve_thinking = False)
    assert drop.preserves_reasoning is False
    asyncio.run(run(drop))
    assert "reasoning" not in sent[-1]["messages"][1]


def test_tool_loop_turns_are_fitted_to_the_engine_window():
    import asyncio

    from routes.managed_engine_chat import _message_fitter

    turn = "word " * 400
    messages = [{"role": "system", "content": "Be brief."}]
    for index in range(10):
        messages += [
            {"role": "user", "content": f"q{index} {turn}"},
            {"role": "assistant", "content": f"a{index} {turn}"},
        ]
    messages.append({"role": "user", "content": "latest"})
    fit = _message_fitter(
        _tool_payload(context_overflow = "truncate_oldest"), _Engine(), "m", tools = None
    )
    fitted, _, line = asyncio.run(fit(messages))
    assert fitted[-1]["content"] == "latest" and len(fitted) < len(messages)
    assert line and "context_truncated" in line
    assert _message_fitter(_tool_payload(), _Engine(), "m", tools = None) is None
    sglang = _Engine()
    sglang.engine = "sglang"
    assert (
        _message_fitter(_tool_payload(context_overflow = "truncate_oldest"), sglang, "m", None)
        is None
    )


def test_tool_loop_honors_the_server_default_overflow(monkeypatch):
    from routes.managed_engine_chat import _message_fitter

    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "truncate_oldest")
    assert _message_fitter(_tool_payload(), _Engine(), "m", tools = None) is not None
    # Auto-compact off in the UI is an explicit "error", which the server default never overrides.
    assert _message_fitter(_tool_payload(context_overflow = "error"), _Engine(), "m", None) is None


def test_a_renamed_deepseek_v31_directory_keeps_its_thinking_toggle(tmp_path):
    template = (
        "{% for m in messages %}<｜User｜>{{ m.content }}{% endfor %}<｜Assistant｜>"
        "{% if thinking is defined and thinking %}<think>{% else %}</think>{% endif %}"
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    (tmp_path / "chat_template.jinja").write_text(template)
    config = SimpleNamespace(is_local = True, path = str(tmp_path), identifier = str(tmp_path))
    options = validate_model(config, engine = "vllm")
    assert options["reasoning_parser"] == "deepseek_v3"
    assert options["reasoning_flags"]["supports_reasoning"] is True
    assert options["reasoning_flags"]["reasoning_always_on"] is False


def test_tokenize_receives_the_effort_inside_the_template_kwargs(monkeypatch):
    # /tokenize has no top-level reasoning_effort, so the count would render the default level.
    from core.inference import managed_engine

    sent = []

    class _Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(
            self,
            url,
            headers = None,
            json = None,
        ):
            sent.append(json)
            return SimpleNamespace(raise_for_status = lambda: None, json = lambda: {"count": 7})

    monkeypatch.setattr(managed_engine.httpx, "Client", _Client)
    engine = ManagedEngine("vllm")
    engine.base_url, engine.model, engine.reasoning_parser = "http://engine", "m", "qwen3"
    assert engine.count_tokens([{"role": "user", "content": "hi"}], reasoning_effort = "low")[0] == 7
    assert "reasoning_effort" not in sent[-1]
    assert sent[-1]["chat_template_kwargs"] == {"reasoning_effort": "low", "enable_thinking": True}
    engine.count_tokens([{"role": "user", "content": "hi"}], enable_thinking = False)
    assert sent[-1]["chat_template_kwargs"] == {"enable_thinking": False, "thinking": False}
