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


@pytest.mark.parametrize("respawn", [False, True])
def test_continuation_evidence_uses_the_remaining_prompt_room(monkeypatch, respawn):
    import httpx
    from .test_document_context_recovery import _usage

    partial = "\n".join(
        f"Finding {i}: condition {i * 37} applies to exemption {i * 13}." for i in range(125)
    )
    payloads = []
    streams = [
        [
            _tool_call_sse("python", {"code": "print('finding')"}, "finding"),
            _usage(5000, 2500),
            _finish("tool_calls"),
            _done(),
        ],
        [
            _sse({"content": "Let me summarize the results."}),
            _usage(5700, 20),
            _finish("stop"),
            _done(),
        ],
        [_sse({"content": partial}), _usage(6712, 1480), _finish("length"), _done()],
        [
            _sse({"content": "\nThe analysis is complete."}),
            _usage(6800, 10),
            _finish("stop"),
            _done(),
        ],
    ]
    if respawn:
        streams.insert(3, httpx.ConnectError("server restarted"))
    backend = _make_backend(monkeypatch, streams, payloads)
    backend._effective_context_length = 10000 if respawn else 8192

    def restart():
        backend._effective_context_length = 8192
        return True

    monkeypatch.setattr(backend, "_respawn_if_dead", restart)
    backend._supports_tools = True

    def count(messages, *args, **kwargs):
        return 500 + sum(len(json.dumps(m)) // 4 for m in messages)

    monkeypatch.setattr(backend, "count_chat_tokens", count)
    monkeypatch.setattr(
        tools, "execute_tool", lambda *a, **kw: "SOURCE FACT: exact exemption.\n" * 80
    )
    summaries = []

    def summary(messages, **kwargs):
        summaries.append(kwargs)
        yield "SOURCE FACT: exact exemption."
        yield {"type": "metadata", "finish_reason": "stop", "usage": {"completion_tokens": 400}}

    monkeypatch.setattr(backend, "generate_chat_completion", summary)
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "user", "content": "Analyze these terms.\n" + "contract terms " * 1200}
            ],
            tools = [tools.PYTHON_TOOL],
            max_tokens = 5000,
            max_tool_iterations = 3,
            context_overflow = "truncate_oldest",
            nudge_tool_calls = True,
        )
    )
    assert len(payloads) == (5 if respawn else 4)
    assert len(summaries) == 1
    assert summaries[0]["max_tokens"] == 500
    final = payloads[-1]["messages"]
    assert 6144 < count(final) <= 7192
    assert any(RECORD_HEADER in m.get("content", "") for m in final)
    assert any("SOURCE FACT: exact exemption." in m.get("content", "") for m in final)
    assert final[-1]["content"] == partial
    assert payloads[-1]["max_tokens"] == 600
    assert events[-1]["usage"]["completion_tokens"] == 4410


@pytest.mark.parametrize("restart_stage", ["initial", "retry"])
@pytest.mark.parametrize("measured_usage", [True, False])
def test_final_answer_respawn_charges_evidence_before_retry(
    monkeypatch, restart_stage, measured_usage
):
    import httpx
    from .test_document_context_recovery import _usage

    payloads = []

    def answer():
        yield _sse({"content": "Completed."})
        yield _usage(300, payloads[-1]["max_tokens"])
        yield _finish("stop")
        yield _done()

    streams = [
        [
            _tool_call_sse("python", {"code": "print('early')"}, "early"),
            _usage(500, 80),
            _finish("tool_calls"),
            _done(),
        ],
        [
            _tool_call_sse("python", {"code": "print('late')"}, "late"),
            _usage(5000, 80),
            _finish("tool_calls"),
            _done(),
        ],
        [
            _tool_call_sse("python", {"code": "print('hidden')"}, "hidden"),
            _usage(6700, 10),
            _finish("tool_calls"),
            _done(),
        ],
        answer(),
    ]
    streams.insert(2 if restart_stage == "initial" else 3, httpx.ConnectError("server restarted"))
    backend = _make_backend(monkeypatch, streams, payloads)
    backend._effective_context_length = 10000
    backend._supports_tools = True
    monkeypatch.setattr(
        backend,
        "count_chat_tokens",
        lambda messages, *a, **kw: 500 + sum(len(json.dumps(m)) // 4 for m in messages),
    )

    def restart():
        backend._effective_context_length = 8192
        return True

    monkeypatch.setattr(backend, "_respawn_if_dead", restart)
    outputs = iter(["EARLY FINDING.\n" * 1200, "LATE FINDING.\n" * 450])
    monkeypatch.setattr(tools, "execute_tool", lambda *a, **kw: next(outputs))
    summaries = []

    def summary(messages, **kwargs):
        summaries.append(kwargs)
        yield "EARLY FINDING."
        if measured_usage:
            yield {"type": "metadata", "finish_reason": "stop", "usage": {"completion_tokens": 600}}

    monkeypatch.setattr(backend, "generate_chat_completion", summary)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Compare every finding."}],
            tools = [tools.PYTHON_TOOL],
            max_tokens = 5000,
            max_tool_iterations = 2,
            context_overflow = "truncate_oldest",
        )
    )
    assert len(payloads) == 5
    assert len(summaries) == 1
    assert "tools" not in payloads[-1]
    retained = any(RECORD_HEADER in m.get("content", "") for m in payloads[-1]["messages"])
    assert retained is measured_usage
    reserved = 0 if measured_usage else summaries[0]["max_tokens"]
    assert payloads[-1]["max_tokens"] == 5000 - 80 - 80 - 10 - (600 if measured_usage else reserved)
    assert events[-1]["usage"]["completion_tokens"] == 5000 - reserved


@pytest.mark.parametrize("outcome", ["timeout", "error", "zero", "timings"])
def test_evidence_reserves_only_unreported_usage(monkeypatch, outcome):
    from core.inference.llama_cpp import _LlamaStreamCancelled
    from .test_document_context_recovery import _usage

    payloads = []
    timers = []

    class Timer:
        def __init__(self, seconds, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(threading, "Timer", Timer)

    def timed_out_summary():
        yield _sse({"content": "An incomplete evidence record."})
        if outcome == "timeout":
            timers[-1].callback()
            raise _LlamaStreamCancelled()
        if outcome == "error":
            raise RuntimeError("summary stream interrupted")
        if outcome == "zero":
            yield _usage(300, 0)
        else:
            yield "data: " + json.dumps({"choices": [], "timings": {"predicted_n": 23}}) + "\n"
        yield _finish("stop")
        yield _done()

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
            timed_out_summary(),
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
    monkeypatch.setattr(
        tools, "execute_tool", lambda *a, **kw: "EARLY FINDING: exact exemption.\n" * 770
    )
    parent_cancel = threading.Event()
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "List every exemption."}],
            tools = [tools.PYTHON_TOOL],
            max_tokens = 5000,
            max_tool_iterations = 3,
            context_overflow = "truncate_oldest",
            cancel_event = parent_cancel,
        )
    )
    assert len(payloads) == 4
    assert len(timers) == 1
    assert not parent_cancel.is_set()
    summary_limit = payloads[2]["max_tokens"]
    assert 0 < summary_limit <= 1914
    reserved = summary_limit if outcome in ("timeout", "error") else 0
    measured = 23 if outcome == "timings" else 0
    assert payloads[-1]["max_tokens"] == 5000 - 80 - 1092 - reserved - measured
    retained = any(RECORD_HEADER in m.get("content", "") for m in payloads[-1]["messages"])
    assert retained is (outcome in ("zero", "timings"))
    assert events[-1]["usage"]["completion_tokens"] == 5000 - reserved


@pytest.mark.parametrize("record_rows", [1, 180])
def test_final_respawn_prices_evidence_with_the_continuation_tail(monkeypatch, record_rows):
    import httpx
    from .test_document_context_recovery import _usage

    partial = "\n".join(
        f"Finding {i}: source {i * 73} establishes exemption {i * 29}." for i in range(250)
    )
    record = "\n".join(
        f"Saved finding {i}: condition {i * 29} applies." for i in range(record_rows)
    )
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
            [_sse({"content": partial}), _usage(6500, 3500), _finish("length"), _done()],
            httpx.ConnectError("server restarted"),
            [
                _sse({"content": "\nAll findings reported."}),
                _usage(7000, 10),
                _finish("stop"),
                _done(),
            ],
        ],
        payloads,
    )
    backend._effective_context_length = 10000
    backend._supports_tools = True

    def count(messages, *args, **kwargs):
        return 500 + sum(len(json.dumps(m)) // 4 for m in messages)

    monkeypatch.setattr(backend, "count_chat_tokens", count)

    def restart():
        backend._effective_context_length = 8192
        return True

    monkeypatch.setattr(backend, "_respawn_if_dead", restart)
    outputs = iter(["EARLY FINDING.\n" * 800, "LATER FINDING.\n" * 730])
    monkeypatch.setattr(tools, "execute_tool", lambda *a, **kw: next(outputs))
    summaries = []

    def summary(messages, **kwargs):
        summaries.append(kwargs)
        yield record
        yield {
            "type": "metadata",
            "finish_reason": "stop",
            "usage": {"completion_tokens": len(record) // 4},
        }

    monkeypatch.setattr(backend, "generate_chat_completion", summary)
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Compare every finding."}],
            tools = [tools.PYTHON_TOOL],
            max_tokens = None,
            max_tool_iterations = 2,
            context_overflow = "truncate_oldest",
        )
    )
    assert len(payloads) == 5
    assert len(summaries) == 1
    final = payloads[-1]["messages"]
    assert final[-1]["content"] == partial
    assert payloads[-1]["continue_final_message"] is True
    assert count(final) + 512 <= 8192, count(final)
    assert any(RECORD_HEADER in m.get("content", "") for m in final) is (record_rows == 1)
    assert "All findings reported." in "".join(
        e.get("text", "") for e in events if e["type"] == "content"
    )


def test_answer_retry_replaces_evidence_before_smaller_window_respawn(monkeypatch):
    import httpx
    from core.inference.llama_cpp import _FINAL_ANSWER_RETRY_NUDGE

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
                _tool_call_sse("python", {"code": "print('hidden')"}, "hidden"),
                _finish("tool_calls"),
                _done(),
            ],
            httpx.ConnectError("server restarted"),
            [_sse({"content": "Completed."}), _finish("stop"), _done()],
        ],
        payloads,
    )
    backend._effective_context_length = 8192
    backend._supports_tools = True
    monkeypatch.setattr(backend, "_final_pass_call_opener_id", lambda: 123)

    def count(messages, *args, **kwargs):
        return 500 + sum(len(json.dumps(m)) // 4 for m in messages)

    monkeypatch.setattr(backend, "count_chat_tokens", count)

    def restart():
        backend._effective_context_length = 4096
        return True

    monkeypatch.setattr(backend, "_respawn_if_dead", restart)
    outputs = iter(
        ["EARLY FINDING: exact exemption.\n" * 500, "LATER FINDING: second exemption.\n" * 200]
    )
    monkeypatch.setattr(tools, "execute_tool", lambda *a, **kw: next(outputs))
    summaries = []
    old_record = "EARLY FINDING: exact exemption.\n" * 60
    new_record = "EARLY AND LATER FINDINGS: merged conditions.\n" * 55

    def summary(messages, **kwargs):
        summaries.append((messages, kwargs))
        record = old_record if len(summaries) == 1 else new_record
        assert len(record) // 4 <= kwargs["max_tokens"]
        yield record
        yield {"type": "metadata", "finish_reason": "stop", "usage": {"completion_tokens": 600}}

    monkeypatch.setattr(backend, "generate_chat_completion", summary)
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [
                {"role": "system", "content": "Preserve exact conditions and sources. " * 125},
                {"role": "user", "content": "List every exemption."},
            ],
            tools = [tools.PYTHON_TOOL],
            max_tokens = None,
            max_tool_iterations = 2,
            context_overflow = "truncate_oldest",
        )
    )
    assert len(payloads) == 5
    assert len(summaries) == 2
    final = payloads[-1]["messages"]
    assert _FINAL_ANSWER_RETRY_NUDGE in str(final)
    assert new_record.strip() in "\n".join(str(m.get("content", "")) for m in final)
    assert sum(str(m.get("content", "")).count(RECORD_HEADER) for m in final) == 1
    assert count(final) <= 3072
    assert "Completed." in "".join(e.get("text", "") for e in events if e["type"] == "content")
