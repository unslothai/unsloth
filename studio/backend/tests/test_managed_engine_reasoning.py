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
        (_QWEN3, "sglang", "qwen3"),
        ("<|channel|>analysis<|message|>", "vllm", "openai_gptoss"),
        ("<|channel|>analysis<|message|>", "sglang", "gpt-oss"),
        ("<｜Assistant｜><think></think>", "vllm", "deepseek_r1"),
        ("<｜Assistant｜><think></think>", "sglang", "deepseek-r1"),
        ("<minimax:tool_call><think></think>", "sglang", "minimax"),
        ("<arg_key><think></think>", "sglang", "glm45"),
        ("<|tool_calls_section_begin|><think></think>", "sglang", "kimi_k2"),
        # V3.1 gates on `thinking` and answers untagged when it is off.
        (
            "<｜Assistant｜>{% if thinking %}<think>{% else %}</think>{% endif %}",
            "vllm",
            "deepseek_v3",
        ),
        (
            "<｜Assistant｜>{% if thinking %}<think>{% else %}</think>{% endif %}",
            "sglang",
            "deepseek-v3",
        ),
        ("<｜Assistant｜>{{ reasoning_effort }}<think></think>", "vllm", None),
        # QwQ / Qwen3 Thinking-2507 open <think> in the prompt, which deepseek_r1 expects.
        ("<|im_start|>assistant\n<think>\n{# </think> #}", "vllm", "deepseek_r1"),
        ("<|im_start|>assistant", "vllm", None),
        (None, "vllm", None),
        ("<|im_start|>assistant", "sglang", None),
    ],
)
def test_reasoning_parser_follows_the_template(template, engine, parser):
    assert reasoning_parser_for_template(template, engine) == parser


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_reasoning_parser_reaches_the_engine_argv(engine):
    argv = ADAPTERS[engine].command(
        "/py", "org/m", 1, "k", 4096, 0.8, options = {"reasoning_parser": "qwen3"}
    )
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert "--reasoning-parser" not in launch_arguments(engine, "/py", "org/m", 1, "k", 4096)


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


def test_sglang_validate_records_its_own_parser_and_the_template_flags(tmp_path):
    options = validate_model(_model_dir(tmp_path, _QWEN3), engine = "sglang")
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


def test_preserved_thinking_reaches_sglang_as_reasoning_content():
    engine = ManagedEngine("sglang")
    engine.reasoning_parser = "qwen3"
    history = [{"role": "assistant", "content": "4", "reasoning_content": "2+2"}]
    assert engine.replay_reasoning(history, True) == history


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
    assert not _managed_engine_unsupported_controls(
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


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_managed_engine_chat_is_fitted_to_the_engine_window(engine):
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {"engine": engine, "context_length": 2048}}
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


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_both_managed_engines_count_the_prompt_with_their_tokenize_route(monkeypatch, engine):
    # SGLang's /tokenize renders messages through its chat template like vLLM's.
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
            sent.append((url, json))
            return SimpleNamespace(raise_for_status = lambda: None, json = lambda: {"count": 11})

    monkeypatch.setattr(managed_engine.httpx, "Client", _Client)
    managed = ManagedEngine(engine)
    managed.base_url, managed.model = "http://engine", "m"
    assert managed.count_tokens([{"role": "user", "content": "hi"}], "Be brief.")[0] == 11
    url, body = sent[-1]
    assert url == "http://engine/tokenize"
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}


def test_a_failed_sglang_count_falls_back_to_an_estimated_fit():
    # SGLang 0.5.17 answers /tokenize 500 for a tokenizer whose model_max_length is the 1e30 sentinel.
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {"engine": "sglang", "context_length": 2048}}
    orchestrator._managed_engine = SimpleNamespace(adapter = ADAPTERS["sglang"])

    def count(*args, **kwargs):
        raise RuntimeError("500 Server Error: Integer exceeds 64-bit range")

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
        messages, context_overflow = "truncate_oldest", max_tokens = 256
    )
    assert fitted["messages"][-1]["content"] == "latest"
    truncation = next(e for e in fitted["events"] if e.get("type") == "context_truncated")
    assert truncation["fits"] and truncation["prompt_tokens_after"] <= 2048 // 2 - 256


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_a_reply_cap_is_clamped_to_the_room_the_prompt_leaves(monkeypatch, engine):
    # Both engines refuse prompt + max_tokens past the window, so a fitted chat with a cap would fail.
    from core.inference import engine_transport

    sent = []
    monkeypatch.setattr(
        engine_transport,
        "stream_chat_events",
        lambda url, headers, payload, cancelled: sent.append(payload) or iter(()),
    )
    managed = ManagedEngine(engine)
    managed.base_url, managed.model, managed.context = "http://engine", "m", 2048
    messages = [{"role": "user", "content": "hi"}]
    for counted, cap, expected in ((1500, 1024, 548), (100, 1024, 1024), (100, 4096, None)):
        managed.count_tokens = lambda *a, counted = counted, **k: (counted, "m")
        list(managed.generate(messages = messages, max_new_tokens = cap))
        assert sent[-1].get("max_tokens") == expected

    def refuse(*a, **k):
        raise RuntimeError("500 Server Error")

    managed.count_tokens = refuse
    list(
        managed.generate(messages = [{"role": "user", "content": "word " * 1250}], max_new_tokens = 256)
    )
    # Without a measured prompt the cap is the engine's to judge; an estimate would cut it to 1.
    assert sent[-1]["max_tokens"] == 256


def test_a_latest_turn_the_strict_estimate_refuses_still_drops_old_history():
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {"engine": "sglang", "context_length": 2048}}
    orchestrator._managed_engine = SimpleNamespace(adapter = ADAPTERS["sglang"])

    def count(*args, **kwargs):
        raise RuntimeError("500 Server Error: Integer exceeds 64-bit range")

    orchestrator.count_chat_tokens = count
    messages = []
    for index in range(10):
        messages += [
            {"role": "user", "content": "word " * 400},
            {"role": "assistant", "content": "ok"},
        ]
    messages.append({"role": "user", "content": "word " * 400 + "latest"})
    fitted = orchestrator.compact_chat_context(
        messages, context_overflow = "truncate_oldest", max_tokens = 256
    )
    assert fitted["messages"][-1]["content"].endswith("latest")
    assert len(fitted["messages"]) < len(messages)


def test_an_estimated_fit_leaves_room_for_the_whole_reply_cap():
    # The cap is sent unchanged when the count fails, so prompt + cap must fit the real window.
    import json

    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.active_model_name = "m"
    orchestrator.models = {"m": {"engine": "sglang", "context_length": 2048}}
    orchestrator._managed_engine = SimpleNamespace(adapter = ADAPTERS["sglang"])

    def count(*args, **kwargs):
        raise RuntimeError("500 Server Error: Integer exceeds 64-bit range")

    orchestrator.count_chat_tokens = count
    messages = []
    for index in range(10):
        messages += [
            {"role": "user", "content": f"question {index}: " + "1,2,3,4,5,6,7,8,9,0," * 70},
            {"role": "assistant", "content": "ok"},
        ]
    messages.append({"role": "user", "content": "Continue."})
    fitted = orchestrator.compact_chat_context(
        messages, context_overflow = "truncate_oldest", max_tokens = 1024
    )
    # A token spans at least one UTF-8 byte, so the serialized bytes bound the real count.
    assert sum(len(json.dumps(m).encode()) for m in fitted["messages"]) + 1024 <= 2048


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
    opens_thought = False
    replay_reasoning = ManagedEngine.replay_reasoning
    continue_final_message = ManagedEngine.continue_final_message


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


def test_tool_loop_continues_only_the_turn_that_ends_on_the_partial():
    import asyncio

    from routes.managed_engine_chat import ManagedToolTransport

    sent = []

    class _Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"reasoning_content":" upon a time"}}]}'

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
    transport = ManagedToolTransport(_Client(), engine, {"model": "m"}, continuing = True)

    async def run(messages):
        lines = transport.stream(
            messages = messages, tools = None, tool_choice = None, cancel_event = asyncio.Event()
        )
        return [json.loads(line[5:])["choices"][0]["delta"] async for line in lines]

    partial = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Once"}]
    assert asyncio.run(run(partial)) == [{"content": " upon a time"}]
    assert sent[-1]["continue_final_message"] is True
    assert sent[-1]["add_generation_prompt"] is False
    later = [*partial, {"role": "tool", "tool_call_id": "c", "content": "r"}]
    assert asyncio.run(run(later)) == [{"reasoning_content": " upon a time"}]
    assert "continue_final_message" not in sent[-1]
    assert "add_generation_prompt" not in sent[-1]


@pytest.mark.parametrize(
    "engine_name,parser",
    [("vllm", "deepseek_r1"), ("vllm", "qwen3"), ("sglang", "deepseek-r1"), ("sglang", "qwen3")],
)
def test_a_continued_answer_stays_answer_under_a_forcing_parser(thinking_peer, engine_name, parser):
    engine, requests = thinking_peer
    engine.engine, engine.reasoning_parser = engine_name, parser
    out = "".join(
        engine.generate(
            messages = [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "The sum is"},
            ],
            continue_final_message = True,
            enable_thinking = True,
        )
    )
    assert out == "two plus two\n\n4"
    assert requests[-1]["continue_final_message"] is True


_R1 = (
    "{% for m in messages %}<｜User｜>{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<｜Assistant｜><think>\n{% endif %}"
)


@pytest.mark.parametrize(
    "template,opens",
    [(_R1, True), (_QWEN3, False), ("<|im_start|>assistant\n", False), ("{% if %}<think>", False)],
)
def test_the_generation_prompt_opening_a_thought_is_read_off_the_template(template, opens):
    from core.inference.managed_engine import generation_prompt_opens_thought
    assert generation_prompt_opens_thought(template) is opens


def test_only_sglang_records_a_generation_prompt_that_opens_a_thought(tmp_path):
    assert validate_model(_model_dir(tmp_path, _R1), engine = "sglang")["opens_thought"] is True
    assert "opens_thought" not in validate_model(_model_dir(tmp_path, _R1), engine = "vllm")
    assert "opens_thought" not in validate_model(_model_dir(tmp_path, _QWEN3), engine = "sglang")


@pytest.mark.parametrize(
    "opens_thought,reasoning,expected",
    [
        # SGLang renders a fresh generation prompt, so R1's open <think> must be closed here.
        (True, " Two plus two. ", "Two plus two.\n</think>\n\nThe sum is"),
        # Raw SGLang text already carries the thought and its close (or is still inside it).
        (True, None, "The sum is"),
        (False, "Two plus two.", "<think>\nTwo plus two.\n</think>\n\nThe sum is"),
        (False, None, "The sum is"),
    ],
)
def test_sglang_prefills_the_finished_thought_before_the_partial(
    thinking_peer, opens_thought, reasoning, expected
):
    engine, requests = thinking_peer
    engine.engine, engine.reasoning_parser, engine.opens_thought = "sglang", None, opens_thought
    partial = {"role": "assistant", "content": "The sum is"}
    if reasoning is not None:
        partial["reasoning_content"] = reasoning
    list(
        engine.generate(
            messages = [{"role": "user", "content": "2+2?"}, partial], continue_final_message = True
        )
    )
    assert requests[-1]["messages"][-1] == {"role": "assistant", "content": expected}
    assert requests[-1]["continue_final_message"] is True


def test_sglang_tool_loop_continuation_carries_the_thought_in_the_prefill():
    import asyncio

    from routes.managed_engine_chat import ManagedToolTransport

    sent = []

    class _Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":" 4"}}]}'

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
    engine.engine, engine.opens_thought = "sglang", True
    engine.base_url, engine.headers = "http://engine", {}
    transport = ManagedToolTransport(_Client(), engine, {"model": "m"}, continuing = True)
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The sum is"}],
            "reasoning_content": "2+2",
        },
    ]

    async def run():
        lines = transport.stream(
            messages = messages, tools = None, tool_choice = None, cancel_event = asyncio.Event()
        )
        return [line async for line in lines]

    asyncio.run(run())
    assert sent[-1]["messages"][-1] == {
        "role": "assistant",
        "content": "2+2\n</think>\n\nThe sum is",
    }
    assert messages[-1]["reasoning_content"] == "2+2"


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
    fit = _message_fitter(_tool_payload(context_overflow = "truncate_oldest"), sglang, "m", None)
    fitted, _, line = asyncio.run(fit(messages))
    assert fitted[-1]["content"] == "latest" and len(fitted) < len(messages)
    assert line and "context_truncated" in line


def test_a_window_sized_reply_cap_is_left_to_the_engine():
    import asyncio

    from routes.managed_engine_chat import _message_fitter

    prompt = [{"role": "user", "content": "word " * 1200}]
    for engine in ("vllm", "sglang"):
        managed = _Engine()
        managed.engine = engine
        for cap, sent in ((None, None), (4096, None), (256, 256)):
            payload = _tool_payload(context_overflow = "truncate_oldest")
            payload.max_tokens = cap
            fit = _message_fitter(payload, managed, "m", None)
            assert asyncio.run(fit(prompt))[1] == sent


def test_a_managed_refit_recalls_with_the_first_branch_it_saw(monkeypatch):
    import asyncio

    from routes import inference as api
    from routes.managed_engine_chat import _message_fitter

    seen = []

    def fake_fit(
        messages,
        payload,
        *,
        tools = None,
        branch = None,
        tool_loop = False,
    ):
        seen.append((list(branch), tool_loop))
        return messages, None, 64

    monkeypatch.setattr(api, "_fit_external_context", fake_fit)
    first = [{"role": "user", "content": "Was there a phone number?"}]
    later = [
        *first,
        {"role": "assistant", "content": "Let me check."},
        {"role": "user", "content": "You described a plan without acting."},
    ]
    fit = _message_fitter(
        _tool_payload(context_overflow = "truncate_oldest"), _Engine(), "m", None, studio_loop = True
    )
    asyncio.run(fit(first))
    asyncio.run(fit(later))
    assert seen == [(first, True), (first, True)]


def test_tool_loop_honors_the_server_default_overflow(monkeypatch):
    from routes.managed_engine_chat import _message_fitter

    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "truncate_oldest")
    assert _message_fitter(_tool_payload(), _Engine(), "m", tools = None) is not None
    # Auto-compact off in the UI is an explicit "error", which the server default never overrides.
    assert _message_fitter(_tool_payload(context_overflow = "error"), _Engine(), "m", None) is None


@pytest.mark.parametrize("engine,parser", [("vllm", "deepseek_v3"), ("sglang", "deepseek-v3")])
def test_a_renamed_deepseek_v31_directory_keeps_its_thinking_toggle(
    tmp_path_factory, engine, parser
):
    template = (
        "{% for m in messages %}<｜User｜>{{ m.content }}{% endfor %}<｜Assistant｜>"
        "{% if thinking is defined and thinking %}<think>{% else %}</think>{% endif %}"
    )
    model = tmp_path_factory.mktemp("my-model")
    (model / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    (model / "chat_template.jinja").write_text(template)
    config = SimpleNamespace(is_local = True, path = str(model), identifier = str(model))
    options = validate_model(config, engine = engine)
    assert options["reasoning_parser"] == parser
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
