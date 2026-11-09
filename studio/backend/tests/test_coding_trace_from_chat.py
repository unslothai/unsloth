# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

import storage.studio_db as studio_db
from core.coding_traces.from_chat import chat_to_trace, load_chat_trace
from core.coding_traces.schema import TRACE_SOURCE, ChatTraceError
from core.external_import import claude, codex, run_import, thread_id_for


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding = "utf-8")


def _claude(
    kind,
    uuid,
    content,
    parent = None,
    second = 0,
):
    return {
        "type": kind,
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": f"2026-08-01T10:00:{second:02d}Z",
        "message": {"role": kind, "content": content},
    }


def _codex(
    kind,
    payload,
    second = 0,
):
    return {"timestamp": f"2026-09-22T12:00:{second:02d}Z", "type": kind, "payload": payload}


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    home = tmp_path / "claude"
    monkeypatch.setenv(claude.SOURCE.home_env, str(home))
    return home


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "codex"
    monkeypatch.setenv(codex.SOURCE.home_env, str(home))
    return home


def _native_thread(tid, messages):
    studio_db.upsert_chat_thread(
        {
            "id": tid,
            "title": "Native",
            "modelType": "base",
            "modelId": "unsloth/model",
            "createdAt": 1,
            "updatedAt": 2,
        }
    )
    studio_db.sync_chat_messages(tid, messages, prune_missing = False)


def _row(mid, role, parts, parent, at):
    return {"id": mid, "role": role, "parentId": parent, "content": parts, "createdAt": at}


def _text(text):
    return [{"type": "text", "text": text}]


def test_claude_import_merges_reply_rows_around_the_tool_result(claude_home):
    _write(
        claude_home / "projects" / "-Users-me-app" / "s1.jsonl",
        [
            _claude("user", "u1", "Fix the test"),
            _claude(
                "assistant",
                "a1",
                [
                    {"type": "text", "text": "Looking."},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}},
                ],
                parent = "u1",
                second = 1,
            ),
            _claude(
                "user",
                "r1",
                [{"type": "tool_result", "tool_use_id": "t1", "content": "a.py"}],
                parent = "a1",
                second = 2,
            ),
            _claude("assistant", "a2", _text("Fixed."), parent = "r1", second = 3),
        ],
    )
    run_import(claude.SOURCE)

    trace = load_chat_trace(thread_id_for(claude.SOURCE, "s1"))

    assert trace.source == TRACE_SOURCE
    assert trace.project_name and "Claude" in trace.project_name
    assert [(m.role, m.text) for m in trace.messages] == [
        ("user", "Fix the test"),
        ("assistant", "Looking."),
        ("tool", "a.py"),
        ("assistant", "Fixed."),
    ]
    call = trace.messages[1].tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("t1", "Bash", {"cmd": "ls"})
    assert trace.messages[2].tool_call_id == "t1"
    assert trace.warnings == []


def test_codex_import_links_function_calls_to_their_output(codex_home):
    _write(
        codex_home / "sessions" / "2026" / "09" / "22" / "rollout-2026-09-22T12-00-00-019a.jsonl",
        [
            _codex("session_meta", {"id": "019a", "cwd": "/Users/me/app", "source": "cli"}),
            _codex("event_msg", {"type": "user_message", "message": "List the files"}),
            _codex(
                "response_item",
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": '{"command": ["ls"]}',
                    "call_id": "c1",
                },
                second = 1,
            ),
            _codex(
                "response_item",
                {"type": "function_call_output", "call_id": "c1", "output": "a.py"},
                second = 2,
            ),
            _codex("event_msg", {"type": "agent_message", "message": "One file: a.py"}, second = 3),
        ],
    )
    run_import(codex.SOURCE)

    trace = load_chat_trace(thread_id_for(codex.SOURCE, "019a"))

    assert [m.role for m in trace.messages] == ["user", "assistant", "tool", "assistant"]
    assert trace.messages[1].tool_calls[0].arguments == {"command": ["ls"]}
    assert trace.messages[2].tool_call_id == "c1"
    assert trace.messages[2].text == "a.py"
    assert trace.messages[3].text == "One file: a.py"
    # Codex keeps its cwd only inside a hashed project id, so no folder reaches the trace.
    assert trace.working_folder is None


def test_native_thread_follows_the_newest_branch_unless_a_head_is_given():
    _native_thread(
        "t1",
        [
            _row("u1", "user", _text("Write a parser"), None, 10),
            _row("a1", "assistant", _text("First try"), "u1", 20),
            _row("a2", "assistant", _text("Second try"), "u1", 30),
            _row("u2", "user", _text("Thanks"), "a1", 40),
        ],
    )

    newest = load_chat_trace("t1")
    assert [m.text for m in newest.messages] == ["Write a parser", "First try", "Thanks"]
    assert newest.model_id == "unsloth/model"
    assert [w.code for w in newest.warnings] == ["other_branches_dropped"]
    assert "Left out 1 " in newest.warnings[0].message

    # A saved head resolves to its newest leaf, the same way the chat UI opens it.
    other = load_chat_trace("t1", head_id = "a2")
    assert [m.text for m in other.messages] == ["Write a parser", "Second try"]
    assert load_chat_trace("t1", head_id = "u1").messages[-1].text == "Thanks"


def test_rows_from_a_service_need_no_database():
    thread = {"id": "t9", "title": "Rows", "modelId": "", "createdAt": 1, "updatedAt": 1}
    messages = [
        _row("u1", "user", _text("Run it") + [{"type": "image", "image": "x"}], None, 1),
        _row(
            "a1",
            "assistant",
            [
                {"type": "reasoning", "text": "plan"},
                {"type": "tool-call", "toolCallId": "k", "toolName": "run", "args": {}},
                {
                    "type": "tool-call",
                    "toolCallId": "k",
                    "toolName": "run",
                    "args": {},
                    "result": {"text": "ok"},
                    "isError": True,
                },
            ],
            "u1",
            2,
        ),
        _row("s1", "tool", _text("stray"), "a1", 3),
    ]

    trace = chat_to_trace(thread, messages)

    assistant, tool = trace.messages[1], trace.messages[2]
    assert assistant.reasoning == "plan"
    assert [c.id for c in assistant.tool_calls] == ["k", "k#2"]
    assert (tool.tool_call_id, tool.text, tool.is_error) == ("k#2", "ok", True)
    assert {w.code for w in trace.warnings} == {
        "unsupported_part",
        "unanswered_tool_call",
        "duplicate_tool_call_id",
        "unknown_role",
    }


def test_a_missing_parent_cuts_the_branch_with_a_warning():
    trace = chat_to_trace(
        {"id": "t"},
        [
            _row("u1", "user", _text("hi"), None, 1),
            _row("a1", "assistant", _text("lost"), "gone", 2),
        ],
    )
    assert [m.text for m in trace.messages] == ["lost"]
    assert {w.code for w in trace.warnings} == {
        "missing_parent",
        "other_branches_dropped",
        "no_user_message",
    }


def test_unreadable_threads_raise():
    with pytest.raises(ChatTraceError, match = "does not exist"):
        load_chat_trace("missing")
    _native_thread("empty", [])
    with pytest.raises(ChatTraceError, match = "no messages"):
        load_chat_trace("empty")
    rows = [_row("u1", "user", _text("hi"), None, 1)]
    with pytest.raises(ChatTraceError, match = "not in this thread"):
        chat_to_trace({"id": "t"}, rows, head_id = "nope")
    with pytest.raises(ChatTraceError, match = "no usable content"):
        chat_to_trace({"id": "t"}, [_row("u1", "user", [{"type": "image"}], None, 1)])
