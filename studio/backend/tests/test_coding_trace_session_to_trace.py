# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path
from unittest import mock

import pytest

import core.coding_traces.session_to_trace as session_to_trace_module
import storage.studio_db as studio_db
from core.coding_traces.session_to_trace import load_session_trace, session_to_trace
from core.coding_traces.schema import TRACE_SOURCE, ChatTraceError
from core.external_import import claude, run_import, thread_id_for
from core.inference.tool_call_parser import TOOL_ERROR_NUDGE


@pytest.fixture(autouse = True)
def no_agents_md(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_AGENTS_MD", "0")


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


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    home = tmp_path / "claude"
    monkeypatch.setenv(claude.SOURCE.home_env, str(home))
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
            "settings": {"systemPrompt": "Be brief."},
        }
    )
    studio_db.sync_chat_messages(tid, messages, prune_missing = False)


def _row(mid, role, parts, parent, at, **extra):
    return {"id": mid, "role": role, "parentId": parent, "content": parts, "createdAt": at, **extra}


def _text(text):
    return [{"type": "text", "text": text}]


def _call(
    cid,
    name = "run",
    result: object = "ok",
    **extra,
):
    part = {"type": "tool-call", "toolCallId": cid, "toolName": name, "args": {}, **extra}
    if result is not None:
        part["result"] = result
    return part


def _assistant(parts):
    return session_to_trace(
        {"id": "t"},
        [_row("u", "user", _text("go"), None, 1), _row("a", "assistant", parts, "u", 2)],
    )


def _shape(trace):
    return [
        (m.role, m.tool_call_id or [c.id for c in m.tool_calls] or m.text) for m in trace.messages
    ]


def _codes(trace):
    return [w.code for w in trace.warnings if w.code != "system_prompt_not_stored"]


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

    trace = load_session_trace(thread_id_for(claude.SOURCE, "s1"))

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


def test_native_thread_follows_the_newest_branch_unless_a_head_is_given():
    _native_thread(
        "t1",
        [
            _row("u1", "user", _text("Write a parser"), None, 10),
            _row("a1", "assistant", _text("First try"), "u1", 50),
            _row("a2", "assistant", _text("Second try"), "u1", 30),
            _row("u2", "user", _text("Thanks"), "a1", 40),
        ],
    )

    newest = load_session_trace("t1")
    assert [m.text for m in newest.messages][1:] == ["Write a parser", "First try", "Thanks"]
    assert newest.model_id == "unsloth/model"
    assert [w.code for w in newest.warnings] == ["other_branches_dropped"]
    assert "Left out 1 saved message(s)" in newest.warnings[0].message

    # A saved head resolves to its newest leaf, the same way the chat UI opens it; the inner node
    # a1 sorts last but never wins.
    other = load_session_trace("t1", head_id = "a2")
    assert [m.text for m in other.messages] == ["Be brief.", "Write a parser", "Second try"]
    assert load_session_trace("t1", head_id = "u1").messages[-1].text == "Thanks"


def test_rows_from_a_service_need_no_database():
    thread = {"id": "t9", "title": "Rows", "modelId": "", "createdAt": 1, "updatedAt": 1}
    raw = {"_raw": "{x"}
    parts = [
        {"type": "reasoning", "text": "plan"},
        _call("k", result = {"text": "ok"}, isError = True, args = raw, argsText = "{x"),
        _call("k", result = {"text": "again"}, args = raw, argsText = "{y"),
        _call("b", args = {"n": 2**53}, argsText = '{"n":9007199254740993}'),
    ]
    messages = [
        _row("u1", "user", _text("Run it") + [{"type": "image", "image": "x"}], None, 1),
        _row("a1", "assistant", parts, "u1", 2),
        _row("s1", "tool", _text("stray"), "a1", 3, attachments = [{"content": _text("file")}]),
    ]

    trace = session_to_trace(thread, messages)
    assistant, tool = trace.messages[1], trace.messages[2]
    assert assistant.reasoning == "plan"
    assert [(c.id, c.arguments) for c in assistant.tool_calls[:2]] == [("k", {}), ("k#2", raw)]
    assert assistant.tool_calls[2].arguments == {"n": 2**53 + 1}  # exact argsText, as the UI sends
    # Byte-exact JSON.stringify, as the UI replays a non-sandbox tool result.
    assert (tool.tool_call_id, tool.text, tool.is_error) == ("k", '{"text":"ok"}', True)
    assert (trace.messages[-1].role, trace.messages[-1].text) == ("assistant", "stray")
    assert _codes(trace) == ["unsupported_part", "duplicate_tool_call_id"]


def test_unreadable_threads_and_cut_branches():
    with pytest.raises(ChatTraceError, match = "does not exist"):
        load_session_trace("missing")
    _native_thread("empty", [])
    with pytest.raises(ChatTraceError, match = "no messages"):
        load_session_trace("empty")
    rows = [_row("u1", "user", _text("hi"), None, 1)]
    with pytest.raises(ChatTraceError, match = "not in this thread"):
        session_to_trace({"id": "t"}, rows, head_id = "nope")
    with pytest.raises(ChatTraceError, match = "no usable content"):
        session_to_trace({"id": "t"}, [_row("u1", "user", [{"type": "image"}], None, 1)])

    trace = session_to_trace(
        {"id": "t"}, rows + [_row("a1", "assistant", _text("lost"), "gone", 2)]
    )
    assert [m.text for m in trace.messages] == ["lost"]
    assert _codes(trace) == ["missing_parent", "other_branches_dropped", "no_user_message"]


def _local(cid, round_id = None):
    return _call(cid, provenance = {"source": "local", "round_id": round_id})


@pytest.mark.parametrize(
    "parts, shape",
    [
        # One row's calls replay as one group past skipped parts; tool messages follow it.
        (
            [_call("A"), {"type": "image", "image": "x"}, _call("B"), *_text("done")],
            [("assistant", ["A", "B"]), ("tool", "A"), ("tool", "B"), ("assistant", "done")],
        ),
        # A new local round closes the group.
        (
            [_local("A", 1), {"type": "thinking", "thinking": "x"}, _local("B", 1), _local("C", 2)],
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
            [_local("A"), _local("B")],
            [("assistant", ["A"]), ("tool", "A"), ("assistant", ["B"]), ("tool", "B")],
        ),
        # Calls without ids use their position.
        (
            [_call(None), _call(None)],
            [("assistant", ["a-0", "a-1"]), ("tool", "a-0"), ("tool", "a-1")],
        ),
    ],
)
def test_tool_calls_group_like_the_ui_replay(parts, shape):
    trace = _assistant(parts)
    assert _shape(trace)[1:] == shape
    assert len({m.id for m in trace.messages}) == len(trace.messages)
    assert set(_codes(trace)) <= {"unsupported_part"}


def test_adjacent_rows_alternate_calls_and_results():
    reasoning = [{"type": "reasoning", "text": t} for t in ("r1", "", "r2")]
    rows = [
        _row("u", "user", _text("go"), None, 1),
        _row("a1", "assistant", [_call("A"), {"type": "source", "url": "http://e"}], "u", 2),
        _row("a2", "assistant", [_call("B")], "a1", 3),
        _row("a3", "assistant", reasoning + _text("done"), "a2", 4),
        _row("a4", "assistant", _text("more"), "a3", 5),
    ]
    trace = session_to_trace({"id": "t"}, rows)
    assert _shape(trace)[1:] == [
        ("assistant", ["A"]),
        ("tool", "A"),
        ("assistant", ["B"]),
        ("tool", "B"),
        ("assistant", "done\nmore"),
    ]
    # A trailing source part must not open a piece that the next row merges into.
    assert (trace.messages[3].id, trace.messages[-1].reasoning) == ("a2", "r1\n\nr2")


@pytest.mark.parametrize(
    "parts, text, codes, note",
    [
        ([*_text("a"), *_text(""), *_text("b")], "a\n\nb", [], None),
        (_text("see [[img:0123456789ab]] now"), "see  now", [], None),
        (_text("clip data:audio/wav;base64,AAAA="), "clip [audio]", [], None),
        (
            [*_text("before"), _call("k", result = None), *_text("after")],
            "before\nafter",
            ["unanswered_tool_call_dropped"],
            "no stored result",
        ),
        (
            [_call("s", "studio_load_skill", None, args = {"name": "lint"}), *_text("ok")],
            "ok",
            ["skill_preload_not_stored"],
            "@lint",
        ),
        (
            [_call("s", "studio_load_skill", None, args = "lint"), *_text("ok")],
            "ok",
            ["skill_preload_not_stored"],
            "@None",
        ),
        (
            [_call("w", "web_search", {"hits": []}, args = {"_server_tool": True}), *_text("ok")],
            "ok",
            ["server_tool_skipped"],
            "'web_search'",
        ),
        (
            [
                _call("x", "code_execution", None, args = {"google": {"native_part": {}}}),
                *_text("ok"),
            ],
            "ok",
            ["server_tool_skipped"],
            "replays its code",
        ),
    ],
)
def test_assistant_rows_replay_like_the_ui(parts, text, codes, note):
    trace = _assistant(parts)
    assert _shape(trace)[1:] == [("assistant", text)]
    assert _codes(trace) == codes
    if note:
        assert note in trace.warnings[0].message


def _turn(custom, content):
    rows = [
        _row("u", "user", _text("go"), None, 1),
        _row("a", "assistant", content, "u", 2, metadata = custom),
        _row("u2", "user", _text("again"), "a", 3),
        _row("a2", "assistant", _text("fine"), "u2", 4),
    ]
    return session_to_trace({"id": "t"}, rows)


_STOPPED = {"incomplete": {"reason": "cancelled"}}


# `before` is what the trace keeps ahead of the next turn ("again", "fine").
@pytest.mark.parametrize(
    "custom, content, before",
    [
        ({}, [], []),
        ({}, _text("   ") + [{"type": "image", "image": ""}], []),
        ({"anthropicRefusal": True}, _text("no"), []),
        (_STOPPED, [], [("user", "go"), ("assistant", "Response stopped")]),
        (_STOPPED, _text("   "), [("user", "go"), ("assistant", "Response stopped")]),
        (
            {"incomplete": {"reason": "length"}},
            [{"type": "reasoning", "text": "hmm"}, _call("k", result = None)],
            [("user", "go"), ("assistant", "Response hit the Max Tokens limit")],
        ),
        # isAbandonedAssistantTurn keeps a turn with a reasoning part, even an empty one.
        ({}, [{"type": "reasoning", "text": ""}], [("user", "go")]),
        # Malformed stored metadata is ignored, as readIncompleteInfo ignores it.
        *[
            (custom, _text("hi"), [("user", "go"), ("assistant", "hi")])
            for custom in ("bad", {"incomplete": "length"}, {"incomplete": {"reason": ["length"]}})
        ],
    ],
)
def test_refused_and_abandoned_turns_are_pruned(custom, content, before):
    assert _shape(_turn(custom, content)) == before + [("user", "again"), ("assistant", "fine")]


def test_stopped_turn_replays_reasoning_beside_text_or_calls():
    # Untrimmed text or a call keeps a stopped group's reasoning; a completed turn always keeps it.
    think = {"type": "reasoning", "text": "think"}
    trace = _turn(_STOPPED, [think, *_text("   ")])
    assert (trace.messages[1].text, trace.messages[1].reasoning) == ("   ", "think")
    assert _shape(_turn(_STOPPED, [_call("k"), think]))[2:4] == [("tool", "k"), ("user", "again")]
    assert _turn({}, [_call("k"), think]).messages[3].reasoning == "think"


@pytest.mark.parametrize("custom", [{}, _STOPPED])
@pytest.mark.parametrize("p", [{"google": {"native_part": {}}}, {"type": "image", "image": "x"}])
def test_native_builtin_or_image_turn_keeps_its_prompt_and_reasoning(custom, p):
    # The UI sends both, so no prune or label; their UI group keeps its reasoning (a call ends one).
    part = p if "image" in p else _call("x", "code_execution", None, args = p)
    trace, tail = _turn(custom, [part]), "" if custom and "image" not in p else "r"
    assert _shape(trace) == [("user", "go"), ("user", "again"), ("assistant", "fine")]
    assert _codes(trace) == ["unsupported_part" if "image" in p else "server_tool_skipped"]
    assert _turn(custom, [{"type": "reasoning", "text": "r"}, part]).messages[1].reasoning == "r"
    assert _turn(custom, [part, {"type": "reasoning", "text": "r"}]).messages[1].reasoning == tail


_NATIVE = _call("x", "code_execution", None, args = {"google": {"native_part": {}}})
_R1, _R2 = {"type": "reasoning", "text": "r1"}, {"type": "reasoning", "text": "r2"}
_IMAGE = {"type": "image", "image": "x"}


@pytest.mark.parametrize(
    "parts, kept",
    [
        ([_R1, _NATIVE, _R2], "r1"),
        ([_R1, *_text("t"), _NATIVE, _R2], "r1"),
        ([*_text("t"), _NATIVE, _R2], ""),
        ([_NATIVE, _R1, _IMAGE], None),
        ([_IMAGE, _R1, _NATIVE, _R2], "r1"),
    ],
)
def test_stopped_turn_decides_reasoning_per_ui_group(parts, kept):
    # A native call ends its UI group, and each group of a stopped turn keeps reasoning alone.
    replayed = {m.id: m.reasoning for m in _turn(_STOPPED, parts).messages}
    assert replayed.get("a") == kept
    assert _turn({}, [p for p in parts if p is not _IMAGE]).messages[1].reasoning  # completed


def test_stopped_row_merged_into_earlier_text_keeps_its_first_group():
    rows = [
        _row("u", "user", _text("go"), None, 1),
        _row("a0", "assistant", _text("hello"), "u", 2),
        _row("a", "assistant", [_R1, _NATIVE, _R2], "a0", 3, metadata = _STOPPED),
    ]
    merged = session_to_trace({"id": "t"}, rows).messages[1]
    assert (merged.text, merged.reasoning) == ("hello", "r1")


@pytest.mark.parametrize("custom", [{}, _STOPPED])
def test_assistant_text_that_python_strips_is_dropped(custom):
    # _build_external_messages drops it. JS trim() keeps "\x1c", so a stopped
    # turn gets no stop label.
    assert _shape(_turn(custom, _text("\x1c"))) == [
        ("user", "go"),
        ("user", "again"),
        ("assistant", "fine"),
    ]


@pytest.mark.parametrize("tool", ["generate_image", "generate_video"])
@pytest.mark.parametrize("text, sent", [("Saved cat.png", "Saved cat.png"), ("", '{"result":""}')])
def test_media_tool_file_wrapper_replays_its_text(tool, text, sent):
    files = [{"name": "cat.png", "size": 10}]
    result = {"text": text, "images": [], "sessionId": "s1", "files": files}
    rows = [
        _row("u", "user", _text("make cat"), None, 1),
        _row("a", "assistant", [_call("c1", tool, result = result), *_text("done")], "u", 2),
    ]
    tool_turn = session_to_trace({"id": "t"}, rows).messages[2]
    assert (tool_turn.role, tool_turn.text) == ("tool", sent)


def test_untrimmed_text_group_after_a_tool_round_is_sent():
    # JS trim() empties "\ufeff" but Python's strip() keeps it, so the backend sends it.
    rows = [
        _row("u", "user", _text("hi"), None, 1),
        _row("a", "assistant", [*_text("go"), _call("c1"), *_text("\ufeff")], "u", 2),
    ]
    assert session_to_trace({"id": "t"}, rows).messages[-1].text == "\ufeff"


def test_stopped_group_with_only_empty_text_drops_its_reasoning():
    parts = [*_text(""), _NATIVE, _R2, *_text("")]
    assert all(m.reasoning != "r2" for m in _turn(_STOPPED, parts).messages)


@pytest.mark.parametrize(
    "part, text",
    [
        (_call("A", "python", {"text": "ok", "sessionId": "s", "images": []}), "ok"),
        (_call("A", "mcp__s__ui", {"text": "", "ui": {"resourceUri": "u"}}), '{"result":""}'),
        (_call("A", "search_knowledge_base", "Hit\n__RAG_SOURCES__:[1]"), "Hit"),
        (_call("A", "python", "Error: boom"), "Error: boom" + TOOL_ERROR_NUDGE),
        (_call("A", result = "out", provenance = {"model_suffix": " [2 left]"}), "out [2 left]"),
        (_call("A", result = "out", provenance = "local"), "out"),
    ],
)
def test_tool_results_match_the_ui_wire_text(part, text):
    wire = json.dumps(part["result"], separators = (",", ":"))
    assert _assistant([part]).messages[2].text == (text or wire)


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
    trace = session_to_trace({"id": "t"}, rows)
    assert [m.text for m in trace.messages] == ["Fix this\ndef f(): pass", "ok", "\ndef f(): pass"]
    assert _codes(trace).count("unsupported_part") == 1


def test_image_only_user_turn_stays_empty():
    rows = [
        _row("u1", "user", _text("hi"), None, 1),
        _row("a1", "assistant", _text("ok"), "u1", 2),
        _row("u2", "user", [{"type": "image", "image": "x"}], "a1", 3),
        _row("a2", "assistant", _text("a cat"), "u2", 4),
    ]
    trace = session_to_trace({"id": "t"}, rows)
    assert _shape(trace) == [
        ("user", "hi"),
        ("assistant", "ok"),
        ("user", ""),
        ("assistant", "a cat"),
    ]
    assert _codes(trace) == ["unsupported_part"]


def test_audio_only_user_turn_stays_between_replies():
    audio = [{"type": "file", "content": [{"type": "audio", "audio": "x"}]}]
    rows = [
        _row("u1", "user", _text("hi"), None, 1),
        _row("a1", "assistant", _text("ok"), "u1", 2),
        _row("u2", "user", _text(""), "a1", 3, attachments = audio),
        _row("a2", "assistant", _text("heard"), "u2", 4),
    ]
    trace = session_to_trace({"id": "t"}, rows)
    assert _shape(trace) == [
        ("user", "hi"),
        ("assistant", "ok"),
        ("user", ""),
        ("assistant", "heard"),
    ]


@pytest.mark.parametrize("parent", [None, "u1"])
def test_legacy_threads_drop_blank_assistant_parts(parent):
    blank = [{"type": "reasoning", "text": " "}, {"type": "image", "image": "https://x/y.png"}]
    joined = _text("A") + _text("  ") + _text("B")
    rows = [
        _row("u1", "user", _text("hi"), None, 1),
        _row("a1", "assistant", blank, "u1" if parent else None, 2),
        _row("u2", "user", _text("again"), "a1" if parent else None, 3),
        _row("a2", "assistant", joined, "u2" if parent else None, 4),
    ]
    trace = session_to_trace({"id": "t"}, rows)
    # fromArray (no parentId anywhere) drops the blank parts, so a1 is abandoned with u1.
    expected = [("user", "again"), ("assistant", "A\nB")]
    if parent:
        expected = [("user", "hi"), ("assistant", ""), ("user", "again"), ("assistant", "A\n  \nB")]
    assert _shape(trace) == expected


def test_legacy_threads_drop_data_images_with_line_breaks():
    image = [{"type": "image", "image": "data:image/png;base64,AA\nBB"}]
    rows = [
        _row("u1", "user", _text("hi"), None, 1),
        _row("a1", "assistant", image, None, 2),
        _row("u2", "user", _text("again"), None, 3),
        _row("a2", "assistant", _text("ok"), None, 4),
    ]
    trace = session_to_trace({"id": "t"}, rows)
    assert _shape(trace) == [("user", "again"), ("assistant", "ok")]


@pytest.mark.parametrize("blank, dropped", [("\ufeff \u3000", True), ("\x1c", False)])
def test_legacy_blank_text_uses_js_trim(blank, dropped):
    rows = [
        _row("u1", "user", _text("hi"), None, 1),
        _row("a1", "assistant", _text(blank), None, 2),
        _row("u2", "user", _text("again"), None, 3),
        _row("a2", "assistant", _text("ok"), None, 4),
    ]
    shape = _shape(session_to_trace({"id": "t"}, rows))
    assert (shape == [("user", "again"), ("assistant", "ok")]) == dropped


def test_infinite_args_text_falls_back_to_args():
    part = _call("A", args = {"a": 1}, argsText = '{"a":1e400}')
    assert _assistant([part]).messages[1].tool_calls[0].arguments == {"a": 1}


def test_deeply_nested_system_variables_fill_nothing():
    deep = "[" * 100000 + "]" * 100000
    thread = {"id": "t", "settings": {"systemPrompt": "{{x}}", "systemVariables": deep}}
    trace = session_to_trace(thread, [_row("u", "user", _text("hi"), None, 1)])
    assert trace.messages[0].text == "{{x}}"


def test_system_variable_numbers_print_like_js():
    variables = '{"a":1e-7,"b":1e21,"c":12345678901234567890,"d":[1e400,0.00001,-0.0],"e":1e400}'
    thread = {"id": "t", "settings": {"systemPrompt": "{{a}} {{b}} {{c}} {{d}} {{e}}"}}
    thread["settings"]["systemVariables"] = variables
    trace = session_to_trace(thread, [_row("u", "user", _text("hi"), None, 1)])
    assert trace.messages[0].text == "1e-7 1e+21 12345678901234567000 [null,0.00001,0] Infinity"


def test_system_variable_objects_print_like_js():
    variables = '{"o":{"b":1,"2":2,"1":3,"01":4,"4294967295":5},"s":{"\\udc00":"\\ud800x"}}'
    thread = {"id": "t", "settings": {"systemPrompt": "{{o}} {{s}}", "systemVariables": variables}}
    trace = session_to_trace(thread, [_row("u", "user", _text("hi"), None, 1)])
    expected = '{"1":3,"2":2,"b":1,"01":4,"4294967295":5} {"\\udc00":"\\ud800x"}'
    assert trace.messages[0].text == expected


def test_system_variable_too_deep_to_print_stays_as_written():
    # The depth where printing runs out of stack depends on the Python build.
    variables = '{"a":[1]}'
    thread = {"id": "t", "settings": {"systemPrompt": "x {{a}} y", "systemVariables": variables}}
    with mock.patch.object(session_to_trace_module, "_js_json", side_effect = RecursionError):
        trace = session_to_trace(thread, [_row("u", "user", _text("hi"), None, 1)])
    assert trace.messages[0].text == "x {{a}} y"


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
            {
                "systemPrompt": "{{repo}} {{a.b}} {{n}} {{ $date }}",
                "systemVariables": '{"repo":"u","a":{"b":"x"},"n":3.0}',
            },
            {"id": "p", "name": "P", "instructions": "Use tabs.", "archived": True},
            "u x 3 {{ $date }}",
            ["system_prompt_variables"],
        ),
        # JSON.parse and String() print 1.0 as 1; a NaN map fails the parse and fills nothing.
        ({"systemPrompt": "{{x}}", "systemVariables": '{"x":[1.0]}'}, None, "[1]", []),
        (
            {"systemPrompt": "{{n}}", "systemVariables": '{"n":NaN}'},
            None,
            "{{n}}",
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
    trace = session_to_trace(thread, [_row("u", "user", _text("hi"), None, 1)], project = project)
    if text is None:
        assert [m.role for m in trace.messages] == ["user"]
    else:
        first = trace.messages[0]
        assert (first.id, first.role, first.text) == ("t:system", "system", text)
    assert [w.code for w in trace.warnings] == codes


def test_agents_md_is_flagged_when_enabled(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_AGENTS_MD")
    assert _codes(_assistant(_text("ok"))) == ["agents_md_not_stored"]
