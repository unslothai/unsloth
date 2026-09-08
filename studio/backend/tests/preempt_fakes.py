# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Doubles shared by the KV-preemption tests."""

from __future__ import annotations

import contextlib
import copy
import json
import threading

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend


def delta(
    content: str,
    *,
    terminator: str = "\n",
    **extra,
) -> str:
    chunk = {"choices": [{"index": 0, "delta": {"content": content}}], **extra}
    return "data: " + json.dumps(chunk) + terminator


def reasoning(content: str, *, terminator: str = "\n") -> str:
    chunk = {"choices": [{"index": 0, "delta": {"reasoning_content": content}}]}
    return "data: " + json.dumps(chunk) + terminator


def finish(reason: str = "stop", *, terminator: str = "\n") -> str:
    chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}
    return "data: " + json.dumps(chunk) + terminator


def done() -> str:
    return "data: [DONE]\n"


def usage(prompt_tokens: int, completion_tokens: int) -> str:
    chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return "data: " + json.dumps(chunk) + "\n"


def tool_call_chunk(
    call_id = "call_search",
    name = "web_search",
    arguments = None,
) -> str:
    call = {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps({"query": "kernel"} if arguments is None else arguments),
        },
    }
    chunk = {"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}
    return "data: " + json.dumps(chunk) + "\n"


def tool_call(call_id: str = "call_search") -> list[str]:
    return [tool_call_chunk(call_id), done()]


def web_search_tool(*, required: bool = False) -> dict:
    parameters: dict = {"type": "object", "properties": {"query": {"type": "string"}}}
    if required:
        parameters["required"] = ["query"]
    return {"type": "function", "function": {"name": "web_search", "parameters": parameters}}


class FakeResponse:
    """Enough of httpx.Response for _iter_text_cancellable."""

    status_code = 200

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False
        self.request = None

    def iter_text(self):
        yield from self._chunks

    def close(self):
        self.closed = True


class RecordingPolicy:
    """Stands in for the admission side. Records the handshake order."""

    def __init__(self, *, resume = True):
        self.events: list[str] = []
        self.checkpoints: list[preemption.StreamCheckpoint] = []
        self._resume = resume

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint) -> None:
        self.events.append("preempted")
        self.checkpoints.append(checkpoint)

    def await_resume(self, timeout = None) -> bool:
        self.events.append("awaited")
        return self._resume

    def on_resumed(self) -> None:
        self.events.append("resumed")


class DecliningPolicy(RecordingPolicy):
    """Knows the current protocol, so a pause it is not given back is recorded."""

    def on_declined(self) -> None:
        self.events.append("declined")


class ServerHookPolicy(RecordingPolicy):
    """Also hears about a park the server took on its own."""

    def on_server_parked(self) -> None:
        self.events.append("server-parked")

    def on_server_resumed(self) -> None:
        self.events.append("server-resumed")


def bare_backend(*, port: int = 48851, **attributes) -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = port
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    for name, value in attributes.items():
        setattr(backend, name, value)
    return backend


class PreemptRecorder:
    """A backend whose upstream stream is scripted and pauses itself where told."""

    def __init__(
        self,
        monkeypatch,
        streams,
        *,
        signal = None,
        pause_attempts = (),
        pause_after = None,
        request_pressure = True,
        patch_iter = True,
        execute_tool = False,
        port = 48851,
        response_factory = None,
        **backend_attributes,
    ):
        self.payloads: list[dict] = []
        self.opened_with_signal: list[object] = []
        self.read_with_signal: list[object] = []
        self.signal = signal
        self.pause_attempts = set(pause_attempts)
        self.pause_after = pause_after
        # attempt index -> data chunks served before the pause. Absent means never.
        if isinstance(pause_after, dict):
            self._plan = dict(pause_after)
        else:
            budget = 1 if pause_after is None else int(pause_after)
            self._plan = {attempt: budget for attempt in self.pause_attempts}
        self._streams = [list(stream) for stream in streams]
        self.backend = bare_backend(port = port, **backend_attributes)
        backend = self.backend
        build_response = response_factory or (
            lambda chunks: type("FakeResponse", (), {"status_code": 200, "chunks": chunks})()
        )

        recorder = self

        @contextlib.contextmanager
        def fake_stream_with_retry(
            _client,
            _url,
            payload,
            _cancel_event,
            headers = None,
            first_token_deadline = None,
            preempt_event = None,
            **_kw,
        ):
            recorder.payloads.append(copy.deepcopy(payload))
            recorder.opened_with_signal.append(preempt_event)
            yield build_response(recorder._streams.pop(0))

        def fake_iter_text_cancellable(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(recorder.payloads) - 1
            recorder.read_with_signal.append(preempt_event)
            budget = recorder._plan.get(attempt)
            served = 0
            for chunk in response.chunks:
                yield chunk
                if not chunk.startswith("data: {"):
                    continue
                served += 1
                if budget is not None and served >= budget:
                    if request_pressure and recorder.signal is not None:
                        recorder.signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        if patch_iter:
            monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
        if execute_tool:
            monkeypatch.setattr(
                "core.inference.tools.execute_tool",
                lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
            )


def run_plain(
    backend,
    *,
    signal,
    policy,
    prompt = "write me a poem",
    **kwargs,
):
    return list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": prompt}],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            **kwargs,
        )
    )


def run_tool_loop(
    backend,
    *,
    signal,
    policy,
    tools,
    prompt = "write me a poem",
    **kwargs,
):
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": prompt}],
            tools = tools,
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            **kwargs,
        )
    )


def _patch_tool_loop(monkeypatch, execute, *, high_risk) -> None:
    from core.inference import studio_tool_loop as loop_mod

    monkeypatch.setattr(loop_mod, "execute_tool", execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", high_risk)


@pytest.fixture
def executed(monkeypatch):
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments})
        return f"RESULT<{name}>"

    _patch_tool_loop(monkeypatch, _execute, high_risk = lambda name, args: False)
    return calls


@pytest.fixture
def rendezvous(monkeypatch):
    # Long enough that a loaded runner meets it, short enough that the serialised cases do not
    # dominate the suite.
    barrier = threading.Barrier(2, timeout = 4)
    order: list[str] = []
    lock = threading.Lock()

    def _execute(name, arguments, **kwargs):
        query = (arguments or {}).get("query", "")
        with lock:
            order.append(query)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return f"ALONE<{query}>"
        return f"TOGETHER<{query}>"

    _patch_tool_loop(monkeypatch, _execute, high_risk = lambda name, args: name == "python")
    return order


@pytest.fixture(autouse = True)
def clean_preemption_registry():
    from core.inference.llama_preemption import reset_preemption_controllers

    reset_preemption_controllers()
    yield
    reset_preemption_controllers()


@pytest.fixture(autouse = True)
def clean_admission_queues():
    from core.inference.llama_admission import reset_llama_admission_queues

    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()
