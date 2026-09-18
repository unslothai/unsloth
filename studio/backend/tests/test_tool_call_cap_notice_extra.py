# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge cases around the over-cap tool-call notice.

The notice is text the model reads, so the load-bearing property is that a turn which
does NOT overflow the cap reads exactly as it did before. These cover that, plus the
per-turn reset, the retry actually executing, the dedup boundary, and
``disable_parallel_tool_use``.
"""

import copy
import json

import pytest

from core.inference.safetensors_agentic import (
    _MAX_TOOL_CALLS_PER_TURN,
    run_safetensors_tool_loop,
)


def _sibling(name):
    """Import a sibling test module. conftest puts the backend root on sys.path, not tests/."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("_cap_notice_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gguf = _sibling("test_llama_cpp_tool_loop")
_safe = _sibling("test_safetensors_tool_loop")

_backend_and_payloads = _gguf._backend_and_payloads
_done = _gguf._done
_record_tool_calls = _gguf._record_tool_calls
_sse = _gguf._sse
FakeExecuteTool = _safe.FakeExecuteTool
_collect_events = _safe._collect_events

NOTICE = "more tool call(s)"


def _blocks(queries):
    return "".join(
        '<tool_call>{"name":"web_search","arguments":%s}</tool_call>' % json.dumps({"query": q})
        for q in queries
    )


def _run_safetensors(
    turn_texts,
    *,
    exec_results = None,
    max_tool_iterations = 4,
):
    """Drive the safetensors loop and return (payloads seen by the model, exec_fn)."""
    seen = []
    turns = iter(turn_texts)

    def _gen(messages):
        seen.append(copy.deepcopy(messages))
        try:
            yield next(turns)
        except StopIteration:
            return

    exec_fn = FakeExecuteTool(exec_results or ["r"] * 64)
    loop = run_safetensors_tool_loop(
        single_turn = _gen,
        messages = [{"role": "user", "content": "hi"}],
        tools = [{"type": "function", "function": {"name": "web_search"}}],
        execute_tool = exec_fn,
        max_tool_iterations = max_tool_iterations,
    )
    _collect_events(loop)
    return seen, exec_fn


def _notices(messages):
    return [m for m in messages if NOTICE in (m.get("content") or "")]


class TestNoticeIsSilentOnNormalTurns:
    """Daniel's question on the PR: does this wording change a normal tool turn?"""

    def test_a_single_call_turn_carries_no_notice(self):
        seen, exec_fn = _run_safetensors([_blocks(["a"]), "final"])

        assert len(exec_fn.calls) == 1
        assert _notices(seen[1]) == []

    def test_a_turn_at_exactly_the_cap_carries_no_notice(self):
        queries = ["q%d" % i for i in range(_MAX_TOOL_CALLS_PER_TURN)]
        seen, exec_fn = _run_safetensors([_blocks(queries), "final"])

        assert len(exec_fn.calls) == _MAX_TOOL_CALLS_PER_TURN
        assert _notices(seen[1]) == []

    def test_gguf_turn_at_exactly_the_cap_carries_no_notice(self, monkeypatch):
        queries = ["q%d" % i for i in range(_MAX_TOOL_CALLS_PER_TURN)]
        streams = [
            [_sse({"content": _blocks(queries)}), _done()],
            [_sse({"content": "done"}), _done()],
        ]
        backend, payloads = _backend_and_payloads(monkeypatch, streams)
        calls = _record_tool_calls(monkeypatch, "OK")

        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "go"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                max_tool_iterations = 2,
            )
        )

        assert len(calls) == _MAX_TOOL_CALLS_PER_TURN
        assert _notices(payloads[1]["messages"]) == []

    def test_raw_calls_over_the_cap_that_dedup_under_it_carry_no_notice(self):
        # 12 raw calls, 4 distinct: the cap is never reached, so this is a dedup turn only.
        queries = ["a", "b", "c", "d"] * 3
        seen, exec_fn = _run_safetensors([_blocks(queries), "final"])

        assert [a["query"] for _name, a in exec_fn.calls] == ["a", "b", "c", "d"]
        assert _notices(seen[1]) == []


class TestNoticeIsPerTurn:
    def test_a_later_normal_turn_does_not_repeat_the_notice(self):
        over = ["q%d" % i for i in range(_MAX_TOOL_CALLS_PER_TURN + 2)]
        seen, _exec_fn = _run_safetensors(
            [_blocks(over), _blocks(["later-1", "later-2"]), "final"],
            max_tool_iterations = 6,
        )

        # Turn 2 is told about the 2 skipped calls; turn 3 inherits that one message from
        # history and gains no new one.
        assert len(_notices(seen[1])) == 1
        assert len(_notices(seen[2])) == 1

    def test_the_skipped_calls_execute_when_the_model_reissues_them(self):
        n = _MAX_TOOL_CALLS_PER_TURN + 2
        over = ["q%d" % i for i in range(n)]
        skipped = over[_MAX_TOOL_CALLS_PER_TURN:]
        _seen, exec_fn = _run_safetensors(
            [_blocks(over), _blocks(skipped), "final"],
            max_tool_iterations = 6,
        )

        # Nothing about the first turn marks the skipped calls as already done, so the
        # retry the notice asks for is actually honoured.
        assert [a["query"] for _name, a in exec_fn.calls] == over
        assert len(exec_fn.calls) == n


class TestDisableParallelToolUse:
    def test_only_one_call_runs_and_no_notice_is_sent(self, monkeypatch):
        queries = ["q%d" % i for i in range(_MAX_TOOL_CALLS_PER_TURN + 2)]
        streams = [
            [_sse({"content": _blocks(queries)}), _done()],
            [_sse({"content": "done"}), _done()],
        ]
        backend, payloads = _backend_and_payloads(monkeypatch, streams)
        calls = _record_tool_calls(monkeypatch, "OK")

        list(
            backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "go"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                max_tool_iterations = 2,
                disable_parallel_tool_use = True,
            )
        )

        assert len(calls) == 1
        # The 9 dropped calls here are dropped by the caller's own setting, not by the
        # per-turn cap, so a cap notice would misattribute them.
        assert _notices(payloads[1]["messages"]) == []


class TestSkippedCallsLeaveNoOrphanUi:
    def test_no_tool_start_or_end_event_for_a_skipped_call(self):
        n = _MAX_TOOL_CALLS_PER_TURN + 2
        queries = ["q%d" % i for i in range(n)]
        turns = iter([_blocks(queries), "final"])

        def _gen(_messages):
            try:
                yield next(turns)
            except StopIteration:
                return

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            execute_tool = FakeExecuteTool(["r"] * n),
            max_tool_iterations = 2,
        )
        events = _collect_events(loop, max_events = 400)

        starts = [e for e in events if e.get("type") == "tool_start"]
        ends = [e for e in events if e.get("type") == "tool_end"]
        assert len(starts) == _MAX_TOOL_CALLS_PER_TURN
        # Every card that opened also closed: a skipped call never leaves a running tool
        # in the UI.
        assert {e.get("tool_call_id") for e in starts} == {e.get("tool_call_id") for e in ends}


@pytest.mark.parametrize("backend", ["gguf", "safetensors", "safetensors_unrestricted"])
@pytest.mark.parametrize("mixed_skipped", [False, True])
@pytest.mark.parametrize("render_result", ["Rendered HTML", "Error: render failed"])
def test_cap_notice_does_not_invite_a_spent_one_shot_retry(
    monkeypatch, backend, mixed_skipped, render_result
):
    calls = [("render_html", {"code": "<p>first</p>"})]
    calls += [("web_search", {"query": f"q{i}"}) for i in range(7)]
    calls.append(("render_html", {"code": "<p>second</p>"}))
    if mixed_skipped:
        calls.append(("web_search", {"query": "later"}))
    text = "".join(
        "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"
        for name, args in calls
    )
    tools = [
        {"type": "function", "function": {"name": name}} for name in ["render_html", "web_search"]
    ]
    if backend == "gguf":
        streams = [[_sse({"content": text}), _done()], [_sse({"content": "done"}), _done()]]
        engine, payloads = _backend_and_payloads(monkeypatch, streams)
        executed = _record_tool_calls(
            monkeypatch, lambda name: render_result if name == "render_html" else "OK"
        )
        list(
            engine.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "go"}],
                tools = tools,
                max_tool_iterations = 3,
            )
        )
        messages = payloads[1]["messages"]
    else:
        payloads = []
        turns = iter([text, "done"])

        def generate(messages):
            payloads.append(copy.deepcopy(messages))
            yield next(turns)

        executor = FakeExecuteTool([render_result] + ["OK"] * 7)
        list(
            run_safetensors_tool_loop(
                single_turn = generate,
                messages = [{"role": "user", "content": "go"}],
                tools = None if backend == "safetensors_unrestricted" else tools,
                execute_tool = executor,
                max_tool_iterations = 3,
            )
        )
        executed = executor.calls
        messages = payloads[1]
    assert len(executed) == 8
    (notice,) = _notices(messages)
    content = notice["content"]
    assert "<p>second</p>" in content
    if render_result.startswith("Error:"):
        assert "Call them again" in content
    else:
        assert "Call them again" not in content
        assert "Do not retry render_html" in content
        if mixed_skipped:
            assert "retry the skipped calls for web_search" in content


def test_skipped_arguments_are_reserved_in_safetensors_result_budgets():
    import random
    import string
    from core.inference.context_window import (
        estimate_messages_tokens_conservative,
        prompt_budget,
    )

    random_source = random.Random(11154)
    alphabet = string.ascii_letters + string.digits
    code = "print(" + json.dumps("".join(random_source.choices(alphabet, k = 4000))) + ")"
    calls = [{"name": "web_search", "arguments": {"query": f"q{i}"}} for i in range(8)]
    calls.append({"name": "python", "arguments": {"code": code}})
    text = "".join("<tool_call>" + json.dumps(call) + "</tool_call>" for call in calls)
    tools = [{"type": "function", "function": {"name": name}} for name in ["web_search", "python"]]
    seen = []
    turns = iter([text, "done"])
    budgets = []

    def generate(messages):
        seen.append(copy.deepcopy(messages))
        yield next(turns)

    def execute(name, arguments, *, result_budget_tokens, **kwargs):
        budgets.append(result_budget_tokens)
        return "".join(random_source.choices(alphabet, k = 2 * max(0, result_budget_tokens - 24)))

    list(
        run_safetensors_tool_loop(
            single_turn = generate,
            messages = [{"role": "user", "content": "go"}],
            tools = tools,
            execute_tool = execute,
            max_tool_iterations = 2,
            context_length = 4096,
            max_tokens = 3500,
        )
    )
    assert len(budgets) == 8
    (notice,) = _notices(seen[1])
    assert json.dumps({"code": code}) in notice["content"]
    spent = estimate_messages_tokens_conservative(seen[1])
    spent += estimate_messages_tokens_conservative(tools)
    assert spent <= prompt_budget(4096, 3500)
