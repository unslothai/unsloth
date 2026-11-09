# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

import storage.studio_db as studio_db
from core.coding_traces.from_chat import chat_to_trace, load_chat_trace
from core.coding_traces.schema import TRACE_SOURCE, ChatTraceError
from core.external_import import claude, codex, cursor, run_import, thread_id_for


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


@pytest.fixture
def cursor_home(tmp_path, monkeypatch):
    home = tmp_path / "cursor"
    monkeypatch.setenv(cursor.SOURCE.home_env, str(home))
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


def _call(
    cid,
    name = "run",
    result = "ok",
    **extra,
):
    part = {"type": "tool-call", "toolCallId": cid, "toolName": name, "args": {}, **extra}
    if result is not None:
        part["result"] = result
    return part


def _assistant(parts):
    return chat_to_trace(
        {"id": "t"},
        [_row("u", "user", _text("go"), None, 1), _row("a", "assistant", parts, "u", 2)],
    )


def _shape(trace):
    return [
        (m.role, m.tool_call_id or [c.id for c in m.tool_calls] or m.text) for m in trace.messages
    ]


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
    assert [w.code for w in trace.warnings] == ["system_prompt_not_stored"]


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
    assert [w.code for w in newest.warnings] == [
        "other_branches_dropped",
        "system_prompt_not_stored",
    ]
    assert "Left out 1 saved message(s)" in newest.warnings[0].message
    assert "other branches" not in newest.warnings[0].message

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
    # Byte-exact JSON.stringify, as the UI replays a non-sandbox tool result.
    assert (tool.tool_call_id, tool.text, tool.is_error) == ("k#2", '{"text":"ok"}', True)
    assert {w.code for w in trace.warnings} == {
        "system_prompt_not_stored",
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
        "system_prompt_not_stored",
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


def test_cursor_import_keeps_calls_without_results_and_warns_once_per_call(cursor_home):
    _write(
        cursor_home / "projects" / "p" / "agent-transcripts" / "s1" / "s1.jsonl",
        [
            {"role": "user", "message": {"content": _text("<user_query>Fix it</user_query>")}},
            {
                "role": "assistant",
                "message": {
                    "content": _text("Looking.")
                    + [{"type": "tool_use", "name": "Read", "input": {"path": "a.py"}}]
                },
            },
            {"role": "assistant", "message": {"content": _text("Fixed.")}},
        ],
    )
    run_import(cursor.SOURCE)

    trace = load_chat_trace(thread_id_for(cursor.SOURCE, "s1"))

    assert [(m.role, m.text) for m in trace.messages] == [
        ("user", "Fix it"),
        ("assistant", "Looking.\n\nFixed."),
    ]
    call = trace.messages[1].tool_calls[0]
    assert (call.name, call.arguments) == ("Read", {"path": "a.py"})
    assert [w.code for w in trace.warnings] == ["unanswered_tool_call", "system_prompt_not_stored"]


def test_a_saved_head_never_resolves_to_an_inner_node():
    rows = [
        _row("u1", "user", _text("Write a parser"), None, 10),
        _row("a1", "assistant", _text("Done"), "u1", 50),
        _row("u2", "user", _text("Add tests"), "a1", 40),
        _row("a2", "assistant", _text("Other"), "u1", 30),
    ]
    trace = chat_to_trace({"id": "t"}, rows, head_id = "u1")
    assert [m.text for m in trace.messages] == ["Write a parser", "Done", "Add tests"]


@pytest.mark.parametrize(
    "parts, shape",
    [
        # One row's calls replay as one group; tool messages follow it before the next text.
        (
            [_call("A"), _call("B"), *_text("done")],
            [("assistant", ["A", "B"]), ("tool", "A"), ("tool", "B"), ("assistant", "done")],
        ),
        # A new local round closes the group.
        (
            [
                _call("A", provenance = {"source": "local", "round_id": 1}),
                _call("B", provenance = {"source": "local", "round_id": 1}),
                _call("C", provenance = {"source": "local", "round_id": 2}),
            ],
            [
                ("assistant", ["A", "B"]),
                ("tool", "A"),
                ("tool", "B"),
                ("assistant", ["C"]),
                ("tool", "C"),
            ],
        ),
        # Completed round-less local calls replay as separate rounds.
        (
            [
                _call("A", provenance = {"source": "local"}),
                _call("B", provenance = {"source": "local"}),
            ],
            [("assistant", ["A"]), ("tool", "A"), ("assistant", ["B"]), ("tool", "B")],
        ),
    ],
)
def test_tool_calls_group_like_the_ui_replay(parts, shape):
    trace = _assistant(parts)
    assert _shape(trace)[1:] == shape
    ids = [m.id for m in trace.messages]
    assert len(ids) == len(set(ids))


def test_adjacent_rows_alternate_calls_and_results():
    rows = [
        _row("u", "user", _text("go"), None, 1),
        _row("a1", "assistant", [_call("A")], "u", 2),
        _row("a2", "assistant", [_call("B")], "a1", 3),
        _row("a3", "assistant", _text("done"), "a2", 4),
    ]
    assert _shape(chat_to_trace({"id": "t"}, rows))[1:] == [
        ("assistant", ["A"]),
        ("tool", "A"),
        ("assistant", ["B"]),
        ("tool", "B"),
        ("assistant", "done"),
    ]


def test_split_assistant_pieces_get_unique_ids():
    trace = _assistant([_call("A"), *_text("x"), _call("B"), *_text("done")])
    assert [m.id for m in trace.messages] == ["u", "a", "a:A", "a#2", "a:B", "a#3"]


def test_calls_without_ids_use_their_position():
    trace = _assistant([_call(None), _call(None)])
    assert [c.id for c in trace.messages[1].tool_calls] == ["a-0", "a-1"]
    assert [m.tool_call_id for m in trace.messages[2:]] == ["a-0", "a-1"]
    assert "duplicate_tool_call_id" not in {w.code for w in trace.warnings}


@pytest.mark.parametrize(
    "result, name, text",
    [
        ("", "run", '{"result":""}'),
        ({"exit_code": 0, "out": "é"}, "run", '{"exit_code":0,"out":"é"}'),
        ({"text": "hi", "score": 3}, "my_client_tool", '{"text":"hi","score":3}'),
        ({"text": "ok", "sessionId": "s", "images": []}, "python", "ok"),
    ],
)
def test_tool_results_match_the_ui_wire_text(result, name, text):
    assert _assistant([_call("A", name, result)]).messages[2].text == text


@pytest.mark.parametrize(
    "part",
    [
        _call("w", "web_search", {"hits": []}, args = {"_server_tool": True}),
        _call("x", "code_execution", None, args = {"google": {"native_part": {}}}),
    ],
)
def test_provider_run_tools_are_skipped(part):
    trace = _assistant([part, *_text("answer")])
    assert _shape(trace)[1:] == [("assistant", "answer")]
    assert [w.code for w in trace.warnings] == ["server_tool_skipped", "system_prompt_not_stored"]


def test_a_client_tool_named_like_a_builtin_is_kept():
    trace = _assistant([_call("w", "web_search", "found")])
    assert _shape(trace)[1:] == [("assistant", ["w"]), ("tool", "w")]


def test_user_attachments_are_model_input():
    attachments = [
        {"content": _text("def f(): pass")},
        {"content": [{"type": "image", "image": "x"}]},
        {"mcpToolOnly": True, "content": _text("secret")},
    ]
    content = _text("Fix this") + [{"type": "text", "text": "hidden", "mcpToolOnly": True}]
    rows = [
        {**_row("u", "user", content, None, 1), "attachments": attachments},
        {**_row("a", "assistant", _text("ok"), "u", 2), "attachments": attachments},
        {**_row("u2", "user", [], "a", 3), "attachments": attachments[:1]},
    ]
    trace = chat_to_trace({"id": "t"}, rows)
    assert [m.text for m in trace.messages] == ["Fix this\n\ndef f(): pass", "ok", "def f(): pass"]
    assert [w.code for w in trace.warnings].count("unsupported_part") == 1


@pytest.mark.parametrize(
    "settings, project, text, codes",
    [
        (
            {"systemPrompt": "  Be brief. "},
            {"id": "p", "name": "P", "instructions": " Use tabs. ", "archived": False},
            "<project_instructions>\nUse tabs.\n</project_instructions>\n\nBe brief.",
            [],
        ),
        (
            {"systemPrompt": "Today is {{$date}}"},
            {"id": "p", "name": "P", "instructions": "Use tabs.", "archived": True},
            "Today is {{$date}}",
            ["system_prompt_variables"],
        ),
        (
            None,
            {"id": "p", "name": "P", "instructions": "Use tabs."},
            "<project_instructions>\nUse tabs.\n</project_instructions>",
            ["system_prompt_not_stored"],
        ),
        ({}, None, None, ["system_prompt_not_stored"]),
    ],
)
def test_the_stored_system_prompt_comes_first(settings, project, text, codes):
    thread = {"id": "t", "settings": settings}
    trace = chat_to_trace(thread, [_row("u", "user", _text("hi"), None, 1)], project = project)
    if text is None:
        assert [m.role for m in trace.messages] == ["user"]
    else:
        assert (trace.messages[0].id, trace.messages[0].role, trace.messages[0].text) == (
            "t:system",
            "system",
            text,
        )
    assert [w.code for w in trace.warnings] == codes


def test_a_native_thread_reads_its_stored_settings():
    studio_db.upsert_chat_thread(
        {"id": "ts", "modelType": "base", "createdAt": 1, "settings": {"systemPrompt": "Be brief."}}
    )
    studio_db.sync_chat_messages(
        "ts", [_row("u", "user", _text("hi"), None, 1)], prune_missing = False
    )
    trace = load_chat_trace("ts")
    assert [(m.role, m.text) for m in trace.messages] == [("system", "Be brief."), ("user", "hi")]
    assert trace.warnings == []
