# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference.llama_cpp import LlamaCppBackend
import routes.inference as inference_route
from .llama_backend_double import FakeLlamaCppBackend

_ANSWER = 'Use re.sub(r"<think>.*?</think>", "", text) to drop the block.'


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _llama_server(monkeypatch, frames):
    real = LlamaCppBackend.__new__(LlamaCppBackend)
    real._process = object()
    real._healthy = True
    real._port = 48847
    real._api_key = None
    real._effective_context_length = 4096
    real._supports_reasoning = True
    real._reasoning_always_on = True
    real._reasoning_style = "enable_thinking"
    real._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream(
        _client,
        _url,
        _payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
        yield type("R", (), {"status_code": 200, "chunks": frames, "close": lambda self: None})()

    monkeypatch.setattr(real, "_stream_with_retry", fake_stream)
    monkeypatch.setattr(
        real,
        "_iter_text_cancellable",
        lambda response, _c, first_token_deadline = None: iter(response.chunks),
    )
    monkeypatch.setattr(real, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    return real


class _ThinkingGguf(FakeLlamaCppBackend):
    reasoning_always_on = True
    supports_reasoning = True
    supports_tools = True

    def __init__(self, real = None):
        self._real = real

    def _request_reasoning_kwargs(self, *args, **kwargs):
        return {}

    def generate_chat_completion(self, **kwargs):
        return self._real.generate_chat_completion(**kwargs)

    def generate_chat_completion_with_tools(self, **kwargs):
        return self._real.generate_chat_completion_with_tools(**kwargs)


def _post(
    monkeypatch,
    backend,
    *,
    stream = True,
    tools = False,
    messages = None,
):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: tools)

    async def _select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    response = TestClient(app).post(
        "/chat/completions",
        json = {
            "messages": messages or [{"role": "user", "content": "hi"}],
            "stream": stream,
            "enable_tools": tools,
            "continue_final_message": messages is not None,
        },
        headers = {"X-Unsloth-Events": "1"} if tools else {},
    )
    assert response.status_code == 200, response.text
    if not stream:
        message = response.json()["choices"][0]["message"]
        return message.get("reasoning_content") or "", message["content"]
    reasoning = content = ""
    for line in response.text.splitlines():
        if not line.startswith("data: {"):
            continue
        for choice in json.loads(line[len("data: ") :]).get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""
    return reasoning, content


def _stream(monkeypatch, frames, **kwargs):
    return _post(monkeypatch, _ThinkingGguf(_llama_server(monkeypatch, frames)), **kwargs)


class _ScriptedToolLoop(_ThinkingGguf):
    """Replays tool-loop events, recording one leading wrap per turn as the backend does."""

    def __init__(self, turns, answer):
        self._turns, self._answer = turns, answer

    def generate_chat_completion_with_tools(self, **kwargs):
        prov = kwargs["reasoning_provenance"]
        for index, trace in enumerate(self._turns):
            if index:
                yield {"type": "status", "text": ""}
            prov.setdefault("wraps", []).append({"len": len(trace)})
            yield {"type": "content", "text": f"<think>{trace}"}
            yield {"type": "content", "text": f"<think>{trace}</think>"}
        yield {"type": "content", "text": f"<think>{self._turns[-1]}</think>{self._answer}"}
        yield {"type": "status", "text": ""}
        yield {"type": "metadata", "finish_reason": "stop"}


def test_literal_think_tags_in_the_answer_stay_answer_text(monkeypatch):
    frames = [_sse({"reasoning_content": "Plan the regex."})]
    frames += [_sse({"content": piece}) for piece in (_ANSWER[:12], _ANSWER[12:30], _ANSWER[30:])]
    frames.append("data: [DONE]\n")

    reasoning, content = _stream(monkeypatch, frames)

    assert reasoning == "Plan the regex."
    assert content == _ANSWER


def test_a_trace_that_quotes_the_closing_tag_stays_reasoning(monkeypatch):
    trace = "I'll match <think> up to </think> lazily."
    frames = [_sse({"reasoning_content": trace[:20]}), _sse({"reasoning_content": trace[20:]})]
    frames += [_sse({"content": "Here it is."}), "data: [DONE]\n"]

    reasoning, content = _stream(monkeypatch, frames)

    assert reasoning == trace
    assert content == "Here it is."


def test_a_reply_cut_off_while_thinking_leaves_no_stray_closing_tag(monkeypatch):
    frames = [_sse({"reasoning_content": "Still thinking about <think> tags"})]
    frames.append(
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        + "\n"
    )
    frames.append("data: [DONE]\n")

    reasoning, content = _stream(monkeypatch, frames)

    assert reasoning == "Still thinking about <think> tags"
    assert content == ""


def test_a_later_reasoning_segment_stays_reasoning(monkeypatch):
    frames = [
        _sse({"reasoning_content": "Plan."}),
        _sse({"content": "Literal <think>x</think> first. "}),
        _sse({"reasoning_content": "Check."}),
        _sse({"content": _ANSWER}),
        "data: [DONE]\n",
    ]

    reasoning, content = _stream(monkeypatch, frames)

    assert reasoning == "Plan.Check."
    assert content == "Literal <think>x</think> first. " + _ANSWER


def test_a_trace_quoting_tool_tags_keeps_the_answer_with_tools_on(monkeypatch):
    trace = "Quote </think> and </function>."
    backend = _ScriptedToolLoop([trace], "OK")

    assert _post(monkeypatch, backend, tools = True) == (trace, "OK")


def test_a_later_reasoning_segment_stays_reasoning_with_tools_on(monkeypatch):
    frames = [
        _sse({"reasoning_content": "Plan."}),
        _sse({"content": "Literal <think>x</think> first. "}),
        _sse({"reasoning_content": "Check."}),
        _sse({"content": "Answer"}),
        "data: [DONE]\n",
    ]

    reasoning, content = _stream(monkeypatch, frames, tools = True)

    assert reasoning == "Plan.Check."
    assert content == "Literal <think>x</think> first. Answer"


def test_a_trace_quoting_a_tool_call_keeps_the_answer_with_tools_on(monkeypatch):
    trace = "Quote </think> and <tool_call>{}</tool_call>."
    frames = [_sse({"reasoning_content": trace}), _sse({"content": "OK"}), "data: [DONE]\n"]

    assert _stream(monkeypatch, frames, tools = True) == (trace, "OK")


def test_a_non_streamed_reply_keeps_literal_tags_where_they_were_written(monkeypatch):
    trace = "Match <think> up to </think> lazily."
    frames = [_sse({"reasoning_content": trace}), _sse({"content": _ANSWER}), "data: [DONE]\n"]

    assert _stream(monkeypatch, frames, stream = False) == (trace, _ANSWER)


def test_a_non_streamed_tool_loop_reply_keeps_literal_tags(monkeypatch):
    trace = "Quote </think> and <tool_call>{}</tool_call>."
    backend = _ScriptedToolLoop(["Search.", trace], _ANSWER)

    assert _post(monkeypatch, backend, stream = False, tools = True) == (trace, _ANSWER)


def _extract(chunks, ledger):
    extractor = inference_route._ResponsesReasoningExtractor(
        parse_think_markers = True, think_provenance = ledger
    )
    parts = [extractor.feed(chunk) for chunk in chunks] + [extractor.finish()]
    return "".join(r for r, _ in parts), "".join(v for _, v in parts)


def test_inline_reasoning_still_parses_when_the_server_split_none_out():
    chunks = ["\n", "<thi", "nk>plan</think>answer"]

    assert _extract(chunks, {"wrapped": 0}) == ("plan", "\nanswer")


def test_a_leading_literal_tag_is_answer_once_the_server_splits_reasoning():
    ledger = {"wrapped": 1, "wraps": [{"len": 5}], "claimed": 1}

    assert _extract(["<think>x</think> is the tag."], ledger) == (
        "",
        "<think>x</think> is the tag.",
    )


def test_reasoning_split_out_after_a_preface_takes_over_from_inline_parsing():
    trace = "Quote </think> literally."
    ledger = {"wrapped": 0}
    extractor = inference_route._ResponsesReasoningExtractor(
        parse_think_markers = True, think_provenance = ledger
    )
    parts = [extractor.feed("Preface. ")]
    ledger["later"] = [{"len": len(trace), "at": len("Preface. ")}]
    parts += [extractor.feed(f"<think>{trace}</think>Use <think>x</think>."), extractor.finish()]

    assert "".join(r for r, _ in parts) == trace
    assert "".join(v for _, v in parts) == "Preface. Use <think>x</think>."


def test_a_continued_answer_after_split_reasoning_keeps_literal_tags(monkeypatch):
    frames = [_sse({"content": '<think>.*?</think>", "", text).'}), "data: [DONE]\n"]
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": 'Use re.sub(r"', "reasoning_content": "Plan."},
    ]

    assert _stream(monkeypatch, frames, messages = messages) == (
        "",
        '<think>.*?</think>", "", text).',
    )
    del messages[-1]["reasoning_content"]
    inline = [_sse({"content": "<think>plan</think>answer"}), "data: [DONE]\n"]
    assert _stream(monkeypatch, inline, messages = messages) == ("plan", "answer")
    # Resuming inside a thought an inline server opened: its closing tag is still a delimiter.
    messages[-1] = {"role": "assistant", "content": "", "reasoning_content": "Plan the regex"}
    closing = [_sse({"content": " carefully.</think>Use regex."}), "data: [DONE]\n"]
    assert _stream(monkeypatch, closing, messages = messages) == ("", " carefully.Use regex.")


def test_a_split_after_an_inline_block_still_claims_its_wrap():
    ledger = {"wrapped": 0}
    extractor = inference_route._ResponsesReasoningExtractor(
        parse_think_markers = True, think_provenance = ledger
    )
    parts = [extractor.feed("<think>Inline.</think>")]
    ledger["later"] = [{"len": len("Split."), "at": len("<think>Inline.</think>")}]
    parts += [extractor.feed("<think>Split.</think>Answer <think>"), extractor.finish()]

    assert "".join(r for r, _ in parts) == "Inline.Split."
    assert "".join(v for _, v in parts) == "Answer <think>"
