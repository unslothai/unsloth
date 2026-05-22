# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the F3 validation-retry pass in ``run_safetensors_tool_loop``.

The pass detects two failure modes between parse and dispatch:

* The model emitted a tool call to a name not in the request's tools.
* The model emitted arguments that are not a JSON object (auto-heal off).

Either failure triggers a corrective tool-result message tied to the
hallucinated ``tool_call_id`` and re-enters the model loop. The pass is
bounded by ``max_validation_retries`` so the loop cannot retry forever.
"""

from core.inference.safetensors_agentic import run_safetensors_tool_loop


class FakeExecuteTool:
    def __init__(self, results = None):
        self.results = list(results or [])
        self.calls = []

    def __call__(self, name, arguments, *, cancel_event = None, timeout = None, session_id = None):
        self.calls.append((name, arguments))
        return self.results.pop(0) if self.results else "OK"


def _multi_turn(turns):
    """Yield cumulative-text generators, one per turn."""
    turn_iter = iter(turns)

    def _gen(_messages):
        try:
            chunks = next(turn_iter)
        except StopIteration:
            return
        acc = ""
        for c in chunks:
            acc += c
            yield acc

    return _gen


def _collect(loop, cap = 200):
    out = []
    for ev in loop:
        out.append(ev)
        if len(out) >= cap:
            break
    return out


REAL_TOOLS = [
    {"type": "function", "function": {"name": "web_search"}},
    {"type": "function", "function": {"name": "python"}},
]


class TestUnknownTool:
    def test_unknown_tool_triggers_retry(self):
        # Turn 1: hallucinated tool name "missing_tool".
        # Turn 2: valid call after the corrective nudge.
        single_turn = _multi_turn([
            ['<tool_call>{"name":"missing_tool","arguments":{}}</tool_call>'],
            ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
            ["Done."],
        ])
        exec_fn = FakeExecuteTool(["result"])
        events = _collect(
            run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = REAL_TOOLS,
                execute_tool = exec_fn,
            )
        )
        # The bad call was NOT executed; the good follow-up call ran.
        assert exec_fn.calls == [("web_search", {"query": "x"})]
        # Final content carries the model's answer.
        contents = [e for e in events if e["type"] == "content"]
        assert any("Done" in e.get("text", "") for e in contents)

    def test_retry_budget_exhausted_falls_through(self):
        # Three unknown calls in a row. With max_validation_retries=2
        # the first two are retried; the third falls through to the
        # existing per-tool error path (which never executes a real
        # tool but emits an error result to the model).
        single_turn = _multi_turn([
            ['<tool_call>{"name":"missing_a","arguments":{}}</tool_call>'],
            ['<tool_call>{"name":"missing_b","arguments":{}}</tool_call>'],
            ['<tool_call>{"name":"missing_c","arguments":{}}</tool_call>'],
            ["Sorry, I cannot proceed."],
        ])
        exec_fn = FakeExecuteTool([])
        events = _collect(
            run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = REAL_TOOLS,
                execute_tool = exec_fn,
                max_validation_retries = 2,
            )
        )
        # execute_tool is never called for an unknown name in either
        # the F3 retry arm or the existing per-tool error arm.
        assert exec_fn.calls == []
        # The loop ultimately exits cleanly; we expect at least one
        # tool_end event from the existing error path on the third turn.
        kinds = [e.get("type") for e in events]
        assert "status" in kinds

    def test_retries_disabled_means_no_retry(self):
        # With max_validation_retries=0, the F3 arm never engages and
        # behavior matches the pre-F3 path: the existing per-tool
        # error message is emitted but no corrective re-entry happens.
        single_turn = _multi_turn([
            ['<tool_call>{"name":"missing","arguments":{}}</tool_call>'],
            ["bye"],
        ])
        exec_fn = FakeExecuteTool([])
        events = _collect(
            run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = REAL_TOOLS,
                execute_tool = exec_fn,
                max_validation_retries = 0,
            )
        )
        # No retry round; the unknown call goes straight to the
        # existing error path.
        assert exec_fn.calls == []
        tool_ends = [e for e in events if e["type"] == "tool_end"]
        # Existing per-tool error path emits a tool_end with an Error
        # result so the model sees the failure.
        assert any(
            "not enabled" in str(e.get("result", ""))
            for e in tool_ends
        )


class TestMalformedArgs:
    def test_malformed_args_bypassed_when_heal_on(self):
        # With auto_heal_tool_calls=True (the default), string args are
        # healed to {"query": "..."} for web_search. F3 leaves heal
        # behavior intact and only catches strictly-impossible shapes.
        single_turn = _multi_turn([
            ['<tool_call>{"name":"web_search","arguments":"some text"}</tool_call>'],
            ["all done"],
        ])
        exec_fn = FakeExecuteTool(["ok"])
        events = _collect(
            run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = REAL_TOOLS,
                execute_tool = exec_fn,
                auto_heal_tool_calls = True,
            )
        )
        # The heal path coerces "some text" to {"query": "some text"}
        # and execute_tool runs.
        assert len(exec_fn.calls) == 1
        assert exec_fn.calls[0][0] == "web_search"

    def test_malformed_args_caught_when_heal_off(self):
        # With auto_heal off, a non-dict arguments value is a hard
        # malformed-args failure and the F3 arm catches it.
        single_turn = _multi_turn([
            ['<tool_call>{"name":"web_search","arguments":"not a dict"}</tool_call>'],
            ['<tool_call>{"name":"web_search","arguments":{"query":"sf"}}</tool_call>'],
            ["sunny"],
        ])
        exec_fn = FakeExecuteTool(["sunny in sf"])
        events = _collect(
            run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = REAL_TOOLS,
                execute_tool = exec_fn,
                auto_heal_tool_calls = False,
            )
        )
        # The first call was caught by F3, not executed. The second
        # (well-formed) call ran.
        assert exec_fn.calls == [("web_search", {"query": "sf"})]


class TestNoOpCases:
    def test_no_tools_no_validation(self):
        # Empty tools list means allowed_tool_names is empty, so the
        # F3 arm never engages.
        single_turn = _multi_turn([
            ['<tool_call>{"name":"anything","arguments":{}}</tool_call>'],
            ["done"],
        ])
        exec_fn = FakeExecuteTool([])
        events = _collect(
            run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [],
                execute_tool = exec_fn,
            )
        )
        # No allowlist gate, no F3 retry; the bad call goes through
        # the dispatch loop and execute_tool runs.
        assert exec_fn.calls == [("anything", {})]

    def test_valid_call_no_retry_overhead(self):
        # A clean valid call does not trigger F3 at all.
        single_turn = _multi_turn([
            ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
            ["final"],
        ])
        exec_fn = FakeExecuteTool(["res"])
        events = _collect(
            run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = REAL_TOOLS,
                execute_tool = exec_fn,
            )
        )
        assert exec_fn.calls == [("web_search", {"query": "x"})]
