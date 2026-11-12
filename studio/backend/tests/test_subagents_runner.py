# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""spawn_subagent: the child runner, the loop branch that starts children in parallel, persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import types

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference import subagents as sa
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools
from storage import studio_db
from storage import subagents_db
from utils.account_context import current_account_id

_DONE = "data: [DONE]"


def _sse(delta = None, finish = None) -> str:
    choice: dict = {"index": 0, "delta": delta if delta is not None else {}}
    if finish is not None:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]})


def _tool_turn(*calls) -> list[str]:
    """One assistant turn with the given (id, name, arguments-dict) tool calls."""
    lines = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call_id,
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
            }
        )
        for index, (call_id, name, arguments) in enumerate(calls)
    ]
    return [*lines, _sse(finish = "tool_calls"), _DONE]


def _answer_turn(text: str) -> list[str]:
    return [_sse({"content": text}), _sse(finish = "stop"), _DONE]


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


WEB = _tool("web_search")
PY = _tool("python")


def _names(tools) -> list[str]:
    return [tool["function"]["name"] for tool in tools or []]


class ScriptedTransport:
    """Replays scripted turns; after them answers [DONE]. ``delay`` is waited out (cancel-aware) per turn."""

    heals_text_tool_calls = False
    sanitizes_provider_frames = True

    def __init__(
        self,
        turns,
        *,
        delay = 0.0,
    ):
        self.turns = [list(turn) for turn in turns]
        self.delay = delay
        self.requests: list[dict] = []
        self.started_at: float | None = None
        self.ended_at: float | None = None

    async def stream(self, messages, tools, tool_choice, cancel_event):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        if self.started_at is None:
            self.started_at = time.monotonic()
        try:
            deadline = time.monotonic() + self.delay
            while time.monotonic() < deadline and not cancel_event.is_set():
                await asyncio.sleep(0.02)
            if cancel_event.is_set():
                return
            for line in self.turns.pop(0) if self.turns else [_DONE]:
                yield line
        finally:
            self.ended_at = time.monotonic()


class AnsweringTransport(ScriptedTransport):
    """Answers ``answer for <brief>`` after ``delay``, whatever the brief is."""

    def __init__(self, delay = 0.0):
        super().__init__([], delay = delay)

    async def stream(self, messages, tools, tool_choice, cancel_event):
        brief = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        self.turns = [_answer_turn(f"answer for {brief}")]
        async for line in super().stream(messages, tools, tool_choice, cancel_event):
            yield line


@pytest.fixture(autouse = True)
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    studio_db.close_wal_keeper()
    studio_db.reset_schema_state_for_tests()
    subagents_db.reset_schema_state_for_tests()
    monkeypatch.setattr(sa, "_runs", {})
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)
    monkeypatch.setattr(
        loop_mod, "strip_result_for_model", lambda result, name = None: result, raising = False
    )
    yield
    studio_db.close_wal_keeper()


@pytest.fixture
def executed(monkeypatch):
    calls: list[dict] = []

    def _execute(name, arguments, **kwargs):
        calls.append({"name": name, "arguments": arguments})
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    return calls


def _policy(
    spawner,
    tools = None,
    **overrides,
) -> ToolLoopPolicy:
    fields = dict(
        tools = tools if tools is not None else [sa.SUBAGENT_TOOL, WEB, PY],
        max_calls = 25,
        timeout = 300,
        permission_mode = "off",
        confirm_calls = False,
        bypass_permissions = False,
        rag_scope = None,
        subagent_spawner = spawner,
    )
    fields.update(overrides)
    return ToolLoopPolicy(**fields)


def _spawner(make_transport, parent_thread = "parent-1") -> sa.SubagentSpawner:
    return sa.SubagentSpawner(
        make_transport = make_transport,
        parent_model = "parent/model",
        parent_thread_id = parent_thread,
        session_id = "sess-1",
    )


def _drive(
    parent,
    spawner,
    *,
    cancel_event = None,
    tools = None,
    on_line = None,
    deadline = 20.0,
    **policy_overrides,
):
    """Run the parent loop to the end, then let every child task finish its saving."""
    cancel_event = cancel_event or threading.Event()

    async def _main():
        lines: list[str] = []
        stream = stream_with_studio_tools(
            parent,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "go"}],
                session_id = "sess-1",
                thread_id = "parent-1",
                model = "parent/model",
            ),
            policy = _policy(spawner, tools, **policy_overrides),
            cancel_event = cancel_event,
        )
        async for line in stream:
            lines.append(line)
            if on_line is not None:
                on_line(line)
        me = asyncio.current_task()
        children = [
            task
            for task in asyncio.all_tasks()
            if task is not me
            and getattr(task.get_coro(), "__qualname__", "") == "SubagentSpawner._run"
        ]
        if children:
            await asyncio.wait(children, timeout = 10)
        return lines

    return asyncio.run(asyncio.wait_for(_main(), timeout = deadline))


def _tool_messages(transport, request_index = 1) -> list[dict]:
    return [m for m in transport.requests[request_index]["messages"] if m.get("role") == "tool"]


def _records() -> dict[str, dict]:
    """Every stored record by description, full."""
    out = {}
    for run in list(sa._runs.values()):
        full = subagents_db.get_subagent(run.record["id"])
        assert full is not None
        out[full["description"]] = full
    return out


def _spawn_args(description, prompt, **extra) -> dict:
    return {"description": description, "prompt": prompt, **extra}


# 1 ------------------------------------------------------------------------------------------


def test_two_spawns_in_one_turn_run_concurrently_and_return_in_call_order(executed):
    parent = ScriptedTransport(
        [
            _tool_turn(
                ("p1", "spawn_subagent", _spawn_args("First task", "alpha")),
                ("p2", "spawn_subagent", _spawn_args("Second task", "beta")),
            ),
            _answer_turn("parent done"),
        ]
    )
    children: list[AnsweringTransport] = []

    def make_transport(model):
        children.append(AnsweringTransport(delay = 0.5))
        return children[-1]

    started = time.monotonic()
    _drive(parent, _spawner(make_transport))
    elapsed = time.monotonic() - started

    assert len(children) == 2
    assert elapsed < 0.9, f"children ran one after the other ({elapsed:.2f}s)"
    first, second = sorted(children, key = lambda c: c.started_at)
    assert second.started_at < first.ended_at, "the two children did not overlap"

    # The parent saw spawn_subagent, the children did not (depth 1).
    assert "spawn_subagent" in _names(parent.requests[0]["tools"])
    for child in children:
        assert "spawn_subagent" not in _names(child.requests[0]["tools"])
        assert _names(child.requests[0]["tools"]) == ["web_search", "python"]
        assert child.requests[0]["messages"][0]["role"] == "system"

    first_result, second_result = [m["content"] for m in _tool_messages(parent)]
    assert first_result == "answer for alpha"
    assert second_result == "answer for beta"

    records = _records()
    assert set(records) == {"First task", "Second task"}
    for description, brief in (("First task", "alpha"), ("Second task", "beta")):
        record = records[description]
        assert record["status"] == "completed"
        assert record["error"] is None
        assert record["resultPreview"] == f"answer for {brief}"
        assert record["parentThreadId"] == "parent-1"
        assert record["sessionId"] == "sess-1"
        assert record["model"] == "parent/model"
        assert record["finishedAt"] >= record["startedAt"]
        assert "spawn_subagent" not in record["tools"]
        assert {"kind": "text", "text": f"answer for {brief}"} in record["transcript"]
    assert executed == []


def test_model_override_goes_to_make_transport(executed):
    parent = ScriptedTransport(
        [
            _tool_turn(
                ("p1", "spawn_subagent", _spawn_args("Cheap job", "x", model = "vendor/small"))
            ),
            _answer_turn("ok"),
        ]
    )
    models: list = []

    def make_transport(model):
        models.append(model)
        return AnsweringTransport()

    _drive(parent, _spawner(make_transport))
    assert models == ["vendor/small"]
    assert _records()["Cheap job"]["model"] == "vendor/small"


# 2 ------------------------------------------------------------------------------------------


def test_child_tool_call_is_recorded_and_tools_argument_filters(executed):
    parent = ScriptedTransport(
        [
            _tool_turn(
                (
                    "p1",
                    "spawn_subagent",
                    _spawn_args("Search things", "find it", tools = ["web_search", "spawn_subagent"]),
                )
            ),
            _answer_turn("parent done"),
        ]
    )
    child = ScriptedTransport(
        [
            _tool_turn(("c1", "web_search", {"query": "cats"})),
            _answer_turn("found cats"),
        ]
    )
    _drive(parent, _spawner(lambda model: child))

    # Only the requested subset, and never spawn_subagent, reached the child.
    assert _names(child.requests[0]["tools"]) == ["web_search"]
    assert executed == [{"name": "web_search", "arguments": {"query": "cats"}}]
    assert _tool_messages(parent)[0]["content"] == "found cats"

    record = _records()["Search things"]
    assert record["tools"] == ["web_search"]
    assert record["toolCallCount"] == 1
    [tool_item] = [item for item in record["transcript"] if item["kind"] == "tool"]
    assert tool_item["toolName"] == "web_search"
    assert tool_item["arguments"] == {"query": "cats"}
    assert tool_item["status"] == "done"
    assert tool_item["result"] == "RESULT<web_search>"
    assert record["transcript"][-1] == {"kind": "text", "text": "found cats"}
    assert record["pendingApproval"] is None


def test_unknown_tools_filter_falls_back_to_all_tools(executed):
    parent = ScriptedTransport(
        [
            _tool_turn(("p1", "spawn_subagent", _spawn_args("Any tools", "x", tools = ["nope"]))),
            _answer_turn("ok"),
        ]
    )
    child = AnsweringTransport()
    _drive(parent, _spawner(lambda model: child))
    assert _names(child.requests[0]["tools"]) == ["web_search", "python"]


# 3 ------------------------------------------------------------------------------------------


def test_parent_cancel_stops_the_running_child(executed):
    parent = ScriptedTransport(
        [_tool_turn(("p1", "spawn_subagent", _spawn_args("Slow task", "wait")))]
    )
    child = AnsweringTransport(delay = 30)
    cancel = threading.Event()

    def on_line(line):
        # The card the loop emits when it starts the call: the child is running now.
        if '"tool_start"' in line and not cancel.is_set():
            threading.Timer(0.3, cancel.set).start()

    started = time.monotonic()
    _drive(parent, _spawner(lambda model: child), cancel_event = cancel, on_line = on_line)
    assert time.monotonic() - started < 10

    record = _records()["Slow task"]
    assert record["status"] == "cancelled"
    assert record["finishedAt"] is not None
    assert sa.live_subagent(record["id"])["status"] == "cancelled"


def test_cancel_subagent_stops_one_child_and_parent_is_told(executed):
    parent = ScriptedTransport(
        [
            _tool_turn(("p1", "spawn_subagent", _spawn_args("Stoppable", "wait"))),
            _answer_turn("parent carried on"),
        ]
    )
    child = AnsweringTransport(delay = 30)
    results: list[bool] = []

    def on_line(line):
        if '"tool_start"' in line and not results:
            results.append(None)

            def _cancel():
                (run,) = sa._runs.values()
                # Unknown ids and finished runs are refused; a running one is stopped.
                assert sa.cancel_subagent("sa_unknown") is False
                results[0] = sa.cancel_subagent(run.record["id"])

            threading.Timer(0.3, _cancel).start()

    _drive(parent, _spawner(lambda model: child), on_line = on_line)

    assert results == [True]
    [content] = [m["content"] for m in _tool_messages(parent)]
    assert "was stopped before it finished" in content
    assert _records()["Stoppable"]["status"] == "cancelled"
    (run,) = sa._runs.values()
    assert sa.cancel_subagent(run.record["id"]) is False


# 4 ------------------------------------------------------------------------------------------


def test_provider_error_fails_the_child_and_the_parent_gets_an_error_result(executed):
    parent = ScriptedTransport(
        [
            _tool_turn(("p1", "spawn_subagent", _spawn_args("Doomed task", "x"))),
            _answer_turn("parent recovered"),
        ]
    )
    child = ScriptedTransport([['data: {"error": {"message": "boom"}}']])
    _drive(parent, _spawner(lambda model: child))

    [content] = [m["content"] for m in _tool_messages(parent)]
    assert content.startswith("Error: subagent")
    assert "boom" in content

    record = _records()["Doomed task"]
    assert record["status"] == "failed"
    assert record["error"] == "boom"
    assert {"kind": "error", "text": "boom"} in record["transcript"]
    assert record["resultPreview"] is None


def test_make_transport_failure_fails_the_child(executed):
    parent = ScriptedTransport(
        [
            _tool_turn(("p1", "spawn_subagent", _spawn_args("No transport", "x"))),
            _answer_turn("ok"),
        ]
    )

    def make_transport(model):
        raise RuntimeError("no connection")

    _drive(parent, _spawner(make_transport))
    [content] = [m["content"] for m in _tool_messages(parent)]
    assert content.startswith("Error: subagent") and "no connection" in content
    assert _records()["No transport"]["status"] == "failed"


# 5 ------------------------------------------------------------------------------------------


def _writer():
    run = sa._Run(
        record = {"transcript": [], "pendingApproval": None, "toolCallCount": 0},
        account_id = current_account_id(),
        cancel_event = threading.Event(),
    )
    return sa._TranscriptWriter(run), run.record


def _event(**payload) -> str:
    return "data: " + json.dumps(payload)


def _delta(**delta) -> str:
    return _sse(delta)


def test_transcript_writer_coalesces_text_and_reasoning():
    writer, record = _writer()
    for line in (
        _delta(reasoning_content = "think "),
        _delta(reasoning_content = "hard"),
        _delta(content = "Hel"),
        _delta(content = "lo"),
        _delta(reasoning = "more"),
        _delta(content = "!"),
        _DONE,
        "event: ping",
        "data: {not json",
    ):
        writer.feed(line)
    assert record["transcript"] == [
        {"kind": "reasoning", "text": "think hard"},
        {"kind": "text", "text": "Hello"},
        {"kind": "reasoning", "text": "more"},
        {"kind": "text", "text": "!"},
    ]
    assert writer.final_answer() == "Hello!"


def test_transcript_writer_approval_lifecycle_and_final_answer_follows_last_tool():
    writer, record = _writer()
    writer.feed(_delta(content = "before tools"))
    writer.feed(_event(type = "tool_status", content = "Running python"))
    assert record["transcript"][-1] == {"kind": "status", "text": "Running python"}
    writer.feed(
        _event(
            type = "tool_start",
            tool_call_id = "t1",
            tool_name = "python",
            arguments = {"code": "1"},
            awaiting_confirmation = True,
            approval_id = "ap1",
        )
    )
    # The transient status badge went away when the card began.
    [text, tool] = record["transcript"]
    assert text["kind"] == "text"
    assert tool["status"] == "awaiting_approval" and tool["approvalId"] == "ap1"
    assert record["pendingApproval"] == {
        "approvalId": "ap1",
        "toolName": "python",
        "toolCallId": "t1",
        "arguments": {"code": "1"},
    }
    writer.feed(_event(type = "tool_approved", tool_call_id = "t1"))
    assert record["pendingApproval"] is None
    assert record["transcript"][-1]["status"] == "running"

    writer.feed(_event(type = "tool_end", tool_call_id = "t1", result = "x" * 5000))
    tool = [i for i in record["transcript"] if i["kind"] == "tool"][0]
    assert tool["status"] == "done"
    assert len(tool["result"]) < 4100 and tool["result"].startswith("xxx")
    assert record["toolCallCount"] == 1

    writer.feed(_delta(content = "the report"))
    assert writer.final_answer() == "the report"


def test_transcript_writer_tool_end_clears_a_pending_approval_and_counts():
    writer, record = _writer()
    writer.feed(
        _event(
            type = "tool_start",
            tool_call_id = "t1",
            tool_name = "python",
            arguments = {},
            awaiting_confirmation = True,
            approval_id = "ap1",
        )
    )
    assert record["pendingApproval"]["approvalId"] == "ap1"
    writer.feed(_event(type = "tool_end", tool_call_id = "t1", result = "ok"))
    assert record["pendingApproval"] is None
    assert record["toolCallCount"] == 1
    assert record["transcript"][0]["result"] == "ok"


def test_transcript_writer_records_provider_errors():
    writer, record = _writer()
    writer.feed(_event(error = {"message": "rate limited"}))
    assert writer.error == "rate limited"
    assert record["transcript"] == [{"kind": "error", "text": "rate limited"}]


# 6 ------------------------------------------------------------------------------------------


def test_finished_child_is_persisted_with_transcript_and_archived_chat(executed, caplog):
    parent = ScriptedTransport(
        [
            _tool_turn(("p1", "spawn_subagent", _spawn_args("Persist me", "do the thing"))),
            _answer_turn("parent done"),
        ]
    )
    child = ScriptedTransport(
        [
            _tool_turn(("c1", "web_search", {"query": "cats"})),
            _answer_turn("all done"),
        ]
    )
    with caplog.at_level(logging.ERROR):
        _drive(parent, _spawner(lambda model: child))

    # _save_child_chat swallows its own errors into the log; a failure must not pass silently.
    failures = [r for r in caplog.records if "save_chat_failed" in r.getMessage()]
    assert not failures, [(r.getMessage(), r.exc_info) for r in failures]
    assert not [r for r in caplog.records if "persist_failed" in r.getMessage()]

    (live,) = sa._runs.values()
    stored = subagents_db.get_subagent(live.record["id"])
    assert stored["status"] == "completed"
    assert stored["pendingApproval"] is None
    assert stored["resultPreview"] == "all done"
    assert stored["toolCallCount"] == 1
    kinds = [item["kind"] for item in stored["transcript"]]
    assert kinds == ["tool", "text"]
    assert [r["id"] for r in subagents_db.list_subagents("parent-1")] == [stored["id"]]
    assert "transcript" not in subagents_db.list_subagents("parent-1")[0]

    thread = studio_db.get_chat_thread(stored["childThreadId"])
    assert thread is not None
    assert thread["title"] == "Subagent · Persist me"
    assert thread["archived"] is True

    messages = studio_db.list_chat_messages(stored["childThreadId"])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == [{"type": "text", "text": "do the thing"}]
    parts = messages[1]["content"]
    assert [part["type"] for part in parts] == ["tool-call", "text"]
    assert parts[0]["toolName"] == "web_search" and parts[0]["result"] == "RESULT<web_search>"
    assert parts[1]["text"] == "all done"
    assert messages[1]["metadata"]["subagent"]["subagentId"] == stored["id"]


def test_registry_is_scoped_to_the_account_and_parent(executed):
    from utils.account_context import AccountContext, run_as

    parent = ScriptedTransport(
        [
            _tool_turn(("p1", "spawn_subagent", _spawn_args("Mine", "x"))),
            _answer_turn("ok"),
        ]
    )
    _drive(parent, _spawner(lambda model: AnsweringTransport()))
    (run,) = sa._runs.values()
    assert [r["id"] for r in sa.live_subagents_for_parent("parent-1")] == [run.record["id"]]
    assert "transcript" not in sa.live_subagents_for_parent("parent-1")[0]
    assert sa.live_subagents_for_parent("other") == []
    assert sa.live_subagent(run.record["id"])["transcript"]
    other = AccountContext("11111111111111111111111111111111", "alice")
    assert run_as(other, sa.live_subagent, run.record["id"]) is None
    assert run_as(other, sa.live_subagents_for_parent, "parent-1") == []
    assert run_as(other, sa.cancel_subagent, run.record["id"]) is False


# Permission sharing ---------------------------------------------------------------------------


def _python_child_turns():
    return [
        _tool_turn(("c1", "python", {"code": "print(1)"})),
        _tool_turn(("c2", "python", {"code": "print(2)"})),
        _answer_turn("ran both"),
    ]


def _spawn_python_child_parent():
    return ScriptedTransport(
        [
            _tool_turn(("p1", "spawn_subagent", _spawn_args("Run python", "run it"))),
            _answer_turn("parent done"),
        ]
    )


@pytest.fixture
def asking(monkeypatch):
    """Python is high-risk; every approval prompt is recorded and answered (optionally with Full access)."""
    from state import tool_approvals as approvals

    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: name == "python")
    state = types.SimpleNamespace(asks = [], full_access = False)

    def answer(slot, approval_id, *args, **kwargs):
        state.asks.append(approval_id)
        assert approvals.resolve_tool_decision(
            approval_id, "allow", session_id = "sess-1", full_access = state.full_access
        )
        return approvals.wait_tool_decision(slot, approval_id, timeout = 0.1)

    monkeypatch.setattr(loop_mod, "wait_tool_decision", answer)
    return state


def _ask_policy(parent_permissions):
    # "auto" prompts only for what is_high_risk flags (python here). In "ask" the parent's own
    # spawn_subagent call prompts too, which would answer (and upgrade) before the child asks anything.
    return dict(
        permission_mode = "auto",
        confirm_calls = True,
        shared_live_permissions = parent_permissions,
    )


def test_child_of_an_upgraded_parent_runs_high_risk_tools_without_asking(executed, asking):
    from state.tool_approvals import LiveToolPermissions

    parent_permissions = LiveToolPermissions("sess-1")
    parent_permissions._full.set()
    child = ScriptedTransport(_python_child_turns())
    _drive(
        _spawn_python_child_parent(),
        _spawner(lambda model: child),
        **_ask_policy(parent_permissions),
    )

    assert asking.asks == []
    assert [call["name"] for call in executed] == ["python", "python"]
    record = _records()["Run python"]
    assert record["status"] == "completed"
    assert record["toolCallCount"] == 2
    assert record["pendingApproval"] is None


def test_child_of_a_non_upgraded_parent_asks_for_approval(executed, asking):
    from state.tool_approvals import LiveToolPermissions

    parent_permissions = LiveToolPermissions("sess-1")
    child = ScriptedTransport(_python_child_turns())
    _drive(
        _spawn_python_child_parent(),
        _spawner(lambda model: child),
        **_ask_policy(parent_permissions),
    )

    # Both calls prompted, and an ordinary Allow upgrades nothing.
    assert len(asking.asks) == 2
    assert [call["name"] for call in executed] == ["python", "python"]
    assert parent_permissions.apply("auto", False) == ("auto", False)
    assert _records()["Run python"]["status"] == "completed"


def test_a_childs_full_access_grant_upgrades_the_parent(executed, asking):
    from state.tool_approvals import LiveToolPermissions

    asking.full_access = True
    parent_permissions = LiveToolPermissions("sess-1")
    child = ScriptedTransport(_python_child_turns())
    _drive(
        _spawn_python_child_parent(),
        _spawner(lambda model: child),
        **_ask_policy(parent_permissions),
    )

    # Asked once; the grant covered the child's later call and now covers the parent too.
    assert len(asking.asks) == 1
    assert [call["name"] for call in executed] == ["python", "python"]
    assert parent_permissions.apply("auto", False) == ("full", True)
    assert _records()["Run python"]["status"] == "completed"


def test_without_a_shared_instance_the_child_gets_its_own_permissions(executed, asking):
    """The parent loop's own instance is what children share, even when the policy carries none."""
    asking.full_access = True
    child = ScriptedTransport(_python_child_turns())
    _drive(
        _spawn_python_child_parent(),
        _spawner(lambda model: child),
        permission_mode = "auto",
        confirm_calls = True,
    )
    assert len(asking.asks) == 1
    assert _records()["Run python"]["status"] == "completed"
