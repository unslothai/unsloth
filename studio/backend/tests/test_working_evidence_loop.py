# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import threading

from core.inference import tools
from core.inference.working_evidence import RECORD_HEADER
from test_llama_cpp_tool_loop import _make_backend, _tool_call_sse, _done, _sse, _finish


def test_evicted_current_findings_reach_final_request_and_usage(monkeypatch):
    payloads = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _tool_call_sse("python", {"code": "print('early')"}, "early"),
                _finish("tool_calls"),
                _done(),
            ],
            [
                _tool_call_sse("python", {"code": "print('late')"}, "late"),
                _finish("tool_calls"),
                _done(),
            ],
            [
                _tool_call_sse("python", {"code": "print('late')"}, "duplicate"),
                _finish("tool_calls"),
                _done(),
            ],
            [_sse({"content": "Completed."}), _finish("stop"), _done()],
        ],
        payloads,
    )
    backend._effective_context_length = 8192
    backend._supports_tools = True

    def count(messages, *args, **kwargs):
        return 500 + sum(len(json.dumps(message)) // 4 for message in messages)

    monkeypatch.setattr(backend, "count_chat_tokens", count)
    outputs = iter(
        ["EARLY FINDING: exact exemption.\n" * 500, "LATER FINDING: second exemption.\n" * 320]
    )
    monkeypatch.setattr(tools, "execute_tool", lambda *args, **kwargs: next(outputs))
    summaries = []
    parent_cancel = threading.Event()

    def summary(messages, **kwargs):
        summaries.append((messages, kwargs))
        yield "EARLY FINDING: exact exemption."
        yield {"type": "metadata", "finish_reason": "stop", "usage": {"completion_tokens": 17}}

    monkeypatch.setattr(backend, "generate_chat_completion", summary)
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "List every exemption."}],
            tools = [tools.PYTHON_TOOL],
            max_tokens = None,
            max_tool_iterations = 4,
            context_overflow = "truncate_oldest",
            cancel_event = parent_cancel,
        )
    )
    assert len(summaries) == 1
    assert "EARLY FINDING" in summaries[0][0][1]["content"]
    final = payloads[-1]["messages"]
    assert any(
        RECORD_HEADER in str(m.get("content")) and "EARLY FINDING" in m["content"] for m in final
    )
    assert any("LATER FINDING" in str(m.get("content")) for m in final)
    assert count(final) <= 6144
    latest_user = next(m for m in reversed(final) if m.get("role") == "user")
    assert RECORD_HEADER in latest_user["content"]
    assert "EARLY FINDING" in latest_user["content"]
    assert any(
        e.get("type") == "metadata" and e["usage"]["completion_tokens"] >= 17 for e in events
    )
    assert summaries[0][1]["_allow_respawn_retry"] is False
    assert summaries[0][1]["enable_thinking"] is False
    parent_cancel.set()
    assert summaries[0][1]["cancel_event"].is_set()
