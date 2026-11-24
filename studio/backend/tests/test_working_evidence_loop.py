# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import threading
import pytest

from core.inference import tools
from core.inference.working_evidence import RECORD_HEADER
from .test_llama_cpp_tool_loop import _make_backend, _tool_call_sse, _done, _sse, _finish


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


@pytest.mark.parametrize("max_tokens", [5000, None, 8192])
def test_retry_cap_accounts_for_preflight_evidence(monkeypatch, max_tokens):
    from .test_document_context_recovery import _usage

    payloads = []

    def answer():
        yield _sse({"content": "Completed."})
        yield _usage(300, payloads[-1]["max_tokens"])
        yield _finish("stop")
        yield _done()

    backend = _make_backend(
        monkeypatch,
        [
            [
                _tool_call_sse("python", {"code": "print('early')"}, "early"),
                _usage(500, 80),
                _finish("tool_calls"),
                _done(),
            ],
            [
                _tool_call_sse("python", {"code": "print('cut')"}, "cut"),
                _usage(7100, 1092),
                _finish("length"),
                _done(),
            ],
            answer(),
        ],
        payloads,
    )
    backend._effective_context_length = 8192
    backend._supports_tools = True
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda messages, *a, **kw: 500 + sum(len(json.dumps(m)) // 4 for m in messages),
    )
    executed = []

    def execute(*args, **kwargs):
        executed.append(args)
        return "EARLY FINDING: exact exemption.\n" * 770

    monkeypatch.setattr(tools, "execute_tool", execute)
    summaries = []

    def summary(messages, **kwargs):
        summaries.append((messages, kwargs))
        yield "EARLY FINDING: exact exemption."
        yield {"type": "metadata", "finish_reason": "stop", "usage": {"completion_tokens": 600}}

    monkeypatch.setattr(backend, "generate_chat_completion", summary)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "List every exemption."}],
            tools = [tools.PYTHON_TOOL],
            max_tokens = max_tokens,
            max_tool_iterations = 3,
            context_overflow = "truncate_oldest",
        )
    )
    assert len(summaries) == 1
    assert len(payloads) == 3
    assert len(executed) == 1
    assert any(RECORD_HEADER in m.get("content", "") for m in payloads[-1]["messages"])
    assert "NONE of that turn" in str(payloads[-1]["messages"])
    if max_tokens == 5000:
        assert payloads[-1]["max_tokens"] == 5000 - 80 - 1092 - 600
        assert events[-1]["usage"]["completion_tokens"] == 5000
    else:
        assert payloads[-1]["max_tokens"] == 8192


def test_working_evidence_does_not_suppress_deferred_history_recall(monkeypatch):
    from core.inference import llama_cpp
    from .test_compact_arguments_before_results import exchange

    historical = "HISTORICAL FACT: previous exemption.\n" * 170
    current = "CURRENT FACT: current exemption.\n" * 375
    code = "print('late')\n" + "# inspect record\n" * 670
    payloads = []
    backend = _make_backend(
        monkeypatch,
        [
            [
                _tool_call_sse("python", {"code": "print('current')"}, "current"),
                _finish("tool_calls"),
                _done(),
            ],
            [_tool_call_sse("python", {"code": code}, "late"), _finish("tool_calls"), _done()],
            [_sse({"content": "Completed comparison."}), _finish("stop"), _done()],
        ],
        payloads,
    )
    backend._effective_context_length = 8192
    backend._supports_tools = True

    def count(messages, *args, **kwargs):
        return 500 + sum(len(json.dumps(m)) // 4 for m in messages)

    monkeypatch.setattr(backend, "count_chat_tokens", count)
    results = iter([current, "LATE FACT: latest exemption."])
    stages = []

    def execute(*args, **kwargs):
        stages.append("execute")
        return next(results)

    monkeypatch.setattr(tools, "execute_tool", execute)

    def summary(messages, **kwargs):
        stages.append("summary")
        assert "CURRENT FACT" in messages[1]["content"]
        assert "HISTORICAL FACT" not in messages[1]["content"]
        yield "CURRENT FACT: current exemption."
        yield {"type": "metadata", "finish_reason": "stop", "usage": {"completion_tokens": 17}}

    monkeypatch.setattr(backend, "generate_chat_completion", summary)
    archived = []
    recall_budgets = []

    def archive(conversation, before, **kwargs):
        result = {
            "conversation": conversation,
            "events": [],
            "counts": {},
            "recalled": False,
            "anchored": [],
        }
        if not kwargs.get("force_recall", True):
            stages.append("archive")
            alive = {id(m) for m in conversation}
            archived.extend(m for m in before if id(m) not in alive)
        elif not kwargs["recall_done"] and archived:
            stages.append("recall")
            recall_budgets.append(kwargs["recall_budget_tokens"])
            assert stages.count("execute") == 2
            recall = {"role": "user", "content": "HISTORICAL FACT: previous exemption."}
            assert count([*conversation, recall]) - count(conversation) <= recall_budgets[-1]
            result.update(conversation = [*conversation, recall], recalled = True, anchored = [recall])
        return result

    monkeypatch.setattr(llama_cpp, "_archive_and_recall", archive)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "Review previous exemptions."},
                *exchange("historical", "print('historical')", historical),
                {
                    "role": "user",
                    "content": "Compare every old exemption with the current document.",
                },
            ],
            tools = [tools.PYTHON_TOOL],
            max_tokens = None,
            max_tool_iterations = 2,
            context_overflow = "truncate_oldest",
        )
    )
    final = "\n".join(str(m.get("content", "")) for m in payloads[-1]["messages"])
    assert RECORD_HEADER in final, stages
    assert "CURRENT FACT: current exemption." in final
    assert any("HISTORICAL FACT" in m.get("content", "") for m in archived), stages
    assert any("CURRENT FACT" in m.get("content", "") for m in archived), stages
    assert stages == ["execute", "archive", "summary", "execute", "recall"]
    assert recall_budgets[0] > 0
    assert "HISTORICAL FACT: previous exemption." in final
    assert "LATE FACT: latest exemption." in final
