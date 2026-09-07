# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Doubles shared by the KV-preemption tests.

Every file that drives a pause through ``LlamaCppBackend`` needs the same three things:
raw SSE chunks to serve, a bare backend whose upstream stream is scripted, and a policy
that records the handshake. Those were written out once per file, which meant a change to
the fake stream signature had to be made in nine places and was twice made in eight.

The parameters here are the differences the tests actually depend on -- which attempt
pauses and after how many chunks, whether the pause raises pressure on the signal, whether
the real cancel-aware reader is left in place. Anything a test does NOT vary is fixed, so
a caller that says nothing gets the ordinary case.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend


# ---------------------------------------------------------------- SSE builders

def delta(content: str, *, terminator: str = "\n", **extra) -> str:
    """One content delta. ``extra`` merges into the chunk, for usage riders and the like."""
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
    """The usage-only chunk llama-server sends last, which a pause is what prevents."""
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
    call_id: str = "call_search",
    name: str = "web_search",
    arguments: dict | None = None,
) -> str:
    """One streamed tool call, as a single delta carrying the whole argument string."""
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    {"query": "kernel"} if arguments is None else arguments
                                ),
                            },
                        }
                    ]
                },
            }
        ]
    }
    return "data: " + json.dumps(chunk) + "\n"


def tool_call(call_id: str = "call_search") -> list[str]:
    """A whole tool-calling round: the call, then the terminator."""
    return [tool_call_chunk(call_id), done()]


def web_search_tool(*, required: bool = False) -> dict:
    parameters: dict = {"type": "object", "properties": {"query": {"type": "string"}}}
    if required:
        parameters["required"] = ["query"]
    return {
        "type": "function",
        "function": {"name": "web_search", "description": "search", "parameters": parameters},
    }


# -------------------------------------------------------------------- responses

class FakeResponse:
    """Enough of httpx.Response for _iter_text_cancellable."""

    status_code = 200

    def __init__(
        self,
        chunks,
        on_close = None,
    ):
        self._chunks = list(chunks)
        self.closed = False
        self._on_close = on_close
        self.request = None

    def iter_text(self):
        yield from self._chunks

    def close(self):
        self.closed = True
        if self._on_close is not None:
            self._on_close()


# --------------------------------------------------------------------- policies

class RecordingPolicy:
    """Stands in for the admission side. Records the handshake order.

    Deliberately has no ``on_declined``: it doubles as the "written against the older
    protocol" case, which the decline path has to survive by way of ``getattr``.
    """

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


# ---------------------------------------------------------------------- backend

def bare_backend(
    *,
    port: int = 48851,
    supports_reasoning: bool = False,
    reasoning_always_on: bool = False,
    **attributes,
) -> LlamaCppBackend:
    """A backend with only the fields the streaming paths read, and no process behind it.

    ``__new__`` rather than the constructor, which would try to launch llama-server.
    """
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = port
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = supports_reasoning
    backend._reasoning_always_on = reasoning_always_on
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    for name, value in attributes.items():
        setattr(backend, name, value)
    return backend


class PreemptRecorder:
    """A backend whose upstream stream is scripted and pauses itself where told.

    ``streams`` is one list of raw SSE chunks per attempt, served in order; ``payloads``
    records what each attempt sent, which is how the tests read what a resume carried
    back. Which attempt pauses and after how many data chunks is ``pause_attempts`` plus
    ``pause_after``: an int applies to every chosen attempt, a dict maps attempt index to
    its own count and is the whole selection, and ``None`` means the first data chunk.
    """

    def __init__(
        self,
        monkeypatch,
        streams,
        *,
        signal = None,
        pause_attempts = (),
        pause_after = None,
        request_pressure: bool = True,
        patch_iter: bool = True,
        execute_tool: bool = False,
        port: int = 48851,
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
                    # Pressure noticed mid-stream, which is when it really is.
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


# ----------------------------------------------------------------------- drivers

def run_plain(backend, *, signal, policy, prompt: str = "write me a poem", **kwargs):
    """Drain a plain streaming chat, which is where a pause has no tool ledger to keep."""
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
    prompt: str = "write me a poem",
    **kwargs,
):
    """Drain the tool loop. Callers that want one round pass ``max_tool_iterations``."""
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
