# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import dataclasses
import gc
import json
import weakref
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import storage.studio_db as studio_db
from auth.authentication import get_current_subject
from core.external_import import (
    claude,
    codex,
    cursor,
    display_name,
    project_id_for,
    run_import,
    thread_id_for,
)
from routes.external_import import router


def _write(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding = "utf-8")
    return path


def _append(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding = "utf-8") as handle:
        handle.write("".join(json.dumps(r) + "\n" for r in records))


def c_user(
    uuid,
    content,
    parent = None,
    ts = "2026-08-01T10:00:00Z",
    **extra,
):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "message": {"role": "user", "content": content},
        **extra,
    }


def c_asst(
    uuid,
    content,
    parent = None,
    ts = "2026-08-01T10:00:01Z",
    **extra,
):
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "message": {"role": "assistant", "content": content},
        **extra,
    }


def k_turn(role, text):
    return {"role": role, "message": {"content": [{"type": "text", "text": text}]}}


def x_line(
    kind,
    payload,
    ts = "2026-09-22T12:00:00Z",
):
    return {"timestamp": ts, "type": kind, "payload": payload}


def _rollout(
    home,
    cwd = "/Users/me/app",
    sid = "019a",
    source = "cli",
    body = None,
    root = "sessions",
    **meta,
):
    path = home / root / "2026" / "09" / "22" / f"rollout-2026-09-22T12-00-00-{sid}.jsonl"
    return _write(
        path,
        [
            x_line(
                "session_meta",
                {"id": sid, "cwd": cwd, "source": source, "originator": "codex_cli_rs", **meta},
            ),
            x_line(
                "response_item",
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<permissions instructions>"}],
                },
            ),
            x_line(
                "response_item",
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "<environment_context>cwd</environment_context>",
                        }
                    ],
                },
            ),
            *(
                body
                if body is not None
                else [
                    x_line("event_msg", {"type": "user_message", "message": "List the files"}),
                    x_line(
                        "response_item",
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "List the files"}],
                        },
                    ),
                    x_line(
                        "response_item",
                        {"type": "reasoning", "summary": [], "encrypted_content": "x"},
                    ),
                    x_line(
                        "response_item",
                        {
                            "type": "function_call",
                            "name": "shell",
                            "arguments": '{"command": ["ls"]}',
                            "call_id": "c1",
                        },
                    ),
                    x_line(
                        "response_item",
                        {"type": "function_call_output", "call_id": "c1", "output": "a.py"},
                        ts = "2026-09-22T12:00:02Z",
                    ),
                    x_line(
                        "response_item",
                        {
                            "type": "custom_tool_call",
                            "name": "apply_patch",
                            "input": "*** Begin Patch",
                            "call_id": "c2",
                        },
                    ),
                    x_line(
                        "response_item",
                        {
                            "type": "custom_tool_call_output",
                            "call_id": "c2",
                            "output": [{"type": "input_text", "text": "Done!"}],
                        },
                    ),
                    x_line(
                        "event_msg",
                        {"type": "agent_message", "message": "One file: a.py"},
                        ts = "2026-09-22T12:00:03Z",
                    ),
                    x_line(
                        "response_item",
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "One file: a.py"}],
                        },
                    ),
                ]
            ),
        ],
    )


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "codex"
    monkeypatch.setenv(codex.SOURCE.home_env, str(home))
    return home


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    home = tmp_path / "claude"
    monkeypatch.setenv(claude.SOURCE.home_env, str(home))
    return home


@pytest.fixture
def cursor_home(tmp_path, monkeypatch):
    home = tmp_path / "cursor"
    monkeypatch.setenv(cursor.SOURCE.home_env, str(home))
    return home


def _session(
    home,
    slug = "-Users-me-app",
    sid = "s1",
    records = None,
):
    return _write(
        home / "projects" / slug / f"{sid}.jsonl",
        records
        or [
            c_user("u1", "Fix the header"),
            c_asst("a1", [{"type": "text", "text": "Fixed it."}], parent = "u1"),
        ],
    )


def _messages(source, sid):
    return studio_db.list_chat_messages(thread_id_for(source, sid))


# Parsing


def test_claude_keeps_the_conversation_and_drops_the_harness(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("m0", "<command-name>/clear</command-name>"),
            c_user("u1", "Fix it\n<local-command-stdout>noise</local-command-stdout>"),
            c_asst(
                "a1",
                [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}},
                ],
                parent = "u1",
            ),
            c_user(
                "r1", [{"type": "tool_result", "tool_use_id": "t1", "content": "a.py"}], parent = "a1"
            ),
            c_asst("a2", [{"type": "text", "text": "Done."}], parent = "r1"),
            c_asst("side", [{"type": "text", "text": "subagent"}], isSidechain = True),
            c_user("meta", "caveat", isMeta = True),
        ],
    )
    t = claude.read_transcript(path, "t", "s1")

    assert [(m["role"], m["content"][0].get("text")) for m in t.messages] == [
        ("user", "Fix it"),
        ("assistant", None),
        ("assistant", "Done."),
    ]
    assert t.messages[1]["content"] == [
        {
            "type": "tool-call",
            "toolCallId": "t1",
            "toolName": "Bash",
            "args": {"cmd": "ls"},
            "result": "a.py",
        }
    ]
    # The tool-result record is skipped, so the answer hangs off the call.
    assert t.messages[2]["parentId"] == t.messages[1]["id"]
    assert t.title == "Fix it"
    assert t.messages[0]["createdAt"] == 1785578400000


JPEG = "/9j/4AAQSkZJRg=="
PNG = "iVBORw0KGgoAAAANSUhEUgAAACgAAAAeCAIAAADRv8uKAAAAK0lEQVR4nO3NMQ0AAAgDsCmZfynIQgYcTfo3056IWCwWi8VisVgsFov/xgvk2cL9h1Oh3gAAAABJRU5ErkJggg=="


def test_claude_keeps_command_requests_and_pasted_images(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("u1", "Where is the login retry loop defined?"),
            c_asst("a1", [{"type": "text", "text": "In auth.py."}], parent = "u1"),
            c_user(
                "cmd",
                "<command-message>explain</command-message>\n<command-name>/explain</command-name>\n"
                "<command-args>how the login retry loop backs off</command-args>",
                parent = "a1",
            ),
            c_user(
                "exp",
                [{"type": "text", "text": "Explain the following in detail: how the login retry"}],
                parent = "cmd",
                isMeta = True,
            ),
            c_asst("a2", [{"type": "text", "text": "It doubles the wait."}], parent = "exp"),
            c_user(
                "cav",
                "<local-command-caveat>The command below was run directly in Claude Code"
                "</local-command-caveat>",
                parent = "a2",
                isMeta = True,
            ),
            c_user(
                "model",
                "<command-name>/model</command-name>\n            <command-message>model"
                "</command-message>\n            <command-args>claude-opus-5-5</command-args>",
                parent = "cav",
            ),
            c_user(
                "out",
                "<local-command-stdout>Set model to `claude-opus-5-5`</local-command-stdout>",
                parent = "model",
            ),
            c_user("clear", "<command-name>/clear</command-name>", parent = "out"),
            c_user(
                "img",
                [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
                    },
                    {"type": "text", "text": "This is the error dialog I get, what does it mean?"},
                ],
                parent = "clear",
            ),
            c_user(
                "path",
                [{"type": "text", "text": "[Image: source: /tmp/images/1.png]"}],
                parent = "img",
                isMeta = True,
            ),
            c_asst("a3", [{"type": "text", "text": "A timeout."}], parent = "path"),
        ],
    )
    messages = claude.read_transcript(path, "t", "s1").messages

    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", [{"type": "text", "text": "Where is the login retry loop defined?"}]),
        ("assistant", [{"type": "text", "text": "In auth.py."}]),
        ("user", [{"type": "text", "text": "/explain how the login retry loop backs off"}]),
        ("assistant", [{"type": "text", "text": "It doubles the wait."}]),
        (
            "user",
            [
                {"type": "image", "image": f"data:image/png;base64,{PNG}"},
                {"type": "text", "text": "This is the error dialog I get, what does it mean?"},
            ],
        ),
        ("assistant", [{"type": "text", "text": "A timeout."}]),
    ]
    ids = [m["id"] for m in messages]
    assert [m["parentId"] for m in messages] == [None, *ids[:-1]]


def _read_image(claude_home, content):
    return _session(
        claude_home,
        records = [
            c_user("u1", "What's wrong in ./shot.png?"),
            c_asst(
                "a1",
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {"file_path": "shot.png"},
                    }
                ],
                parent = "u1",
            ),
            c_user(
                "r1",
                [{"type": "tool_result", "tool_use_id": "t1", "content": content}],
                parent = "a1",
            ),
            c_asst("a2", [{"type": "text", "text": "The disk is full."}], parent = "r1"),
        ],
    )


IMAGE_RESULT = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": PNG},
}
PNG_IMAGE = {"data": PNG, "mimeType": "image/png"}
JPEG_RESULT = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/"},
}


@pytest.mark.parametrize(
    "content, text, images",
    [
        ([IMAGE_RESULT], "", [PNG_IMAGE]),
        (
            [
                {"type": "text", "text": "Page 1"},
                IMAGE_RESULT,
                {"type": "image", "source": "malformed"},
                JPEG_RESULT,
            ],
            "Page 1",
            [PNG_IMAGE, {"data": "/9j/", "mimeType": "image/jpeg"}],
        ),
    ],
)
def test_claude_keeps_images_a_tool_returned(claude_home, content, text, images):
    _read_image(claude_home, content)
    run_import(claude.SOURCE)

    call = _messages(claude.SOURCE, "s1")[1]["content"][0]
    assert call["result"] == {"text": text, "images": images}


def test_claude_parallel_tool_images_reach_their_own_calls(claude_home):
    def use(tool_id):
        return {"type": "tool_use", "id": tool_id, "name": "Read", "input": {}}

    def result(tool_id, image):
        return {"type": "tool_result", "tool_use_id": tool_id, "content": [image]}

    path = _session(
        claude_home,
        records = [
            c_user("u1", "Compare them"),
            c_asst("a1", [use("t1"), use("t2")], parent = "u1"),
            c_user("r1", [result("t2", JPEG_RESULT), result("t1", IMAGE_RESULT)], parent = "a1"),
        ],
    )
    calls = claude.read_transcript(path, "t", "s1").messages[1]["content"]
    assert {c["toolCallId"]: c["result"]["images"][0]["mimeType"] for c in calls} == {
        "t1": "image/png",
        "t2": "image/jpeg",
    }


def test_claude_rewinds_keep_their_branch(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            c_asst("a1", [{"type": "text", "text": "A"}], parent = "u1"),
            c_user("u2", "two", parent = "a1"),
            c_user("u2b", "two again", parent = "a1"),
        ],
    )
    ids = [m["id"] for m in claude.read_transcript(path, "t", "s1").messages]
    parents = [m["parentId"] for m in claude.read_transcript(path, "t", "s1").messages]
    assert parents == [None, ids[0], ids[1], ids[1]]


def test_claude_harness_records_and_compaction_keep_one_conversation(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            c_user("meta", "caveat", parent = "u1", isMeta = True),
            {"type": "attachment", "uuid": "at1", "parentUuid": "meta", "attachment": {}},
            c_asst("a1", [{"type": "text", "text": "A"}], parent = "at1"),
            {"type": "system", "subtype": "turn_duration", "uuid": "s1", "parentUuid": "a1"},
            c_user("u2", "two", parent = "s1"),
            c_asst("a2", [{"type": "text", "text": "B"}], parent = "u2"),
            {
                "type": "system",
                "subtype": "compact_boundary",
                "uuid": "cb",
                "parentUuid": None,
                "logicalParentUuid": "a2",
            },
            c_user("u3", "three", parent = "cb"),
            c_asst("a3", [{"type": "text", "text": "C"}], parent = "u3"),
        ],
    )
    messages = claude.read_transcript(path, "t", "s1").messages
    ids = [m["id"] for m in messages]
    assert [m["parentId"] for m in messages] == [None, *ids[:-1]]


def test_claude_rewind_compaction_uses_its_logical_parent(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            c_asst("a1", [{"type": "text", "text": "A"}], parent = "u1"),
            c_user("u2", "abandoned", parent = "a1"),
            c_asst("a2", [{"type": "text", "text": "B"}], parent = "u2"),
            {
                "type": "system",
                "subtype": "compact_boundary",
                "uuid": "cb",
                "parentUuid": None,
                "logicalParentUuid": "a1",
            },
            c_user("sum", "summary", parent = "cb", isCompactSummary = True),
            c_user("u3", "three", parent = "sum"),
            c_asst("a3", [{"type": "text", "text": "C"}], parent = "u3"),
        ],
    )
    messages = claude.read_transcript(path, "t", "s1").messages
    ids = [m["id"] for m in messages]
    assert [m["parentId"] for m in messages] == [
        None,
        ids[0],
        ids[1],
        ids[2],
        ids[1],
        ids[4],
        ids[5],
    ]


def test_claude_releases_filtered_records_while_streaming(tmp_path, monkeypatch):
    class Record(dict):
        pass

    released = False

    def records(_path):
        nonlocal released
        attachment = Record(type = "attachment", uuid = "at1", parentUuid = None, payload = "x" * 1_000_000)
        reference = weakref.ref(attachment, lambda _ref: mark_released())
        yield attachment
        attachment = None
        yield {"type": "progress", "uuid": "p1", "parentUuid": "at1"}
        gc.collect()
        assert reference() is None
        yield c_user("u1", "one", parent = "p1")

    def mark_released():
        nonlocal released
        released = True

    path = tmp_path / "session.jsonl"
    path.touch()
    monkeypatch.setattr(claude, "read_jsonl", records)
    messages = claude.read_transcript(path, "t", "s1").messages
    assert released
    assert len(messages) == 1


def test_claude_duplicate_reply_ids_do_not_merge_sequential_responses(claude_home):
    def block(uuid, reply, tool_id, parent):
        record = c_asst(
            uuid,
            [{"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"cmd": "ls"}}],
            parent = parent,
        )
        record["message"]["id"] = reply
        return record

    def result(uuid, tool_id, parent):
        return c_user(
            uuid, [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}], parent = parent
        )

    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            block("a1", "duplicate", "t1", "u1"),
            result("r1", "t1", "a1"),
            block("a2", "duplicate", "t2", "r1"),
            result("r2", "t2", "a2"),
            c_asst("a3", [{"type": "text", "text": "done"}], parent = "r2"),
            c_user("u2", "rewind", parent = "a1"),
        ],
    )
    messages = claude.read_transcript(path, "t", "s1").messages
    ids = [m["id"] for m in messages]
    assert [m["parentId"] for m in messages] == [None, ids[0], ids[1], ids[2], ids[1]]


def test_claude_duplicate_reply_ids_do_not_merge_sibling_responses(claude_home):
    first = c_asst("a1", [{"type": "text", "text": "first"}], parent = "u1")
    first["message"]["id"] = "duplicate"
    alternate = c_asst("a2", [{"type": "text", "text": "alternate"}], parent = "u1")
    alternate["message"]["id"] = "duplicate"
    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            first,
            alternate,
            c_user("u2", "rewind from first", parent = "a1"),
        ],
    )
    messages = claude.read_transcript(path, "t", "s1").messages
    ids = [m["id"] for m in messages]
    assert [m["parentId"] for m in messages] == [None, ids[0], ids[0], ids[1]]


def test_claude_duplicate_hook_uuids_keep_per_occurrence_ancestry(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            {"type": "saved_hook_context", "uuid": "hook", "parentUuid": "u1"},
            c_asst("a1", [{"type": "text", "text": "A"}], parent = "hook"),
            c_user("u2", "two", parent = "a1"),
            {"type": "saved_hook_context", "uuid": "hook", "parentUuid": "u2"},
            c_asst("a2", [{"type": "text", "text": "B"}], parent = "hook"),
        ],
    )
    messages = claude.read_transcript(path, "t", "s1").messages
    ids = [m["id"] for m in messages]
    assert [m["parentId"] for m in messages] == [None, ids[0], ids[1], ids[2]]


@pytest.mark.parametrize("logical", ["never-written", "at2"])
def test_claude_parallel_tool_calls_and_compaction_keep_one_conversation(claude_home, logical):
    def block(uuid, reply, content, parent):
        record = c_asst(uuid, content, parent = parent)
        record["message"]["id"] = reply
        return record

    def call(tool_id):
        return [{"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"cmd": "ls"}}]

    def result(uuid, tool_id, parent):
        return c_user(
            uuid, [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}], parent = parent
        )

    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            block("a1", "m1", call("t1"), "u1"),
            block("a1b", "m1", call("t2"), "a1"),
            result("r2", "t2", "a1b"),
            result("r1", "t1", "a1"),
            block("a2", "m2", [{"type": "text", "text": "A"}], "r1"),
            c_user("u2", "/compact", parent = "a2"),
            {
                "type": "system",
                "subtype": "compact_boundary",
                "uuid": "cb",
                "parentUuid": None,
                "logicalParentUuid": logical,
            },
            c_user("sum", "summary", parent = "cb", isCompactSummary = True),
            {"type": "attachment", "uuid": "at2", "parentUuid": "sum", "attachment": {}},
            c_user("u3", "three", parent = "at2"),
            c_asst("a3", [{"type": "text", "text": "C"}], parent = "u3"),
        ],
    )
    messages = claude.read_transcript(path, "t", "s1").messages
    ids = [m["id"] for m in messages]
    assert len(messages) == 8
    assert [m["parentId"] for m in messages] == [None, *ids[:-1]]


def test_claude_null_message_inside_a_streamed_reply_keeps_the_branch(claude_home):
    def block(uuid, tool_id, parent):
        record = c_asst(
            uuid, [{"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}}], parent = parent
        )
        record["message"]["id"] = "m1"
        return record

    def result(uuid, tool_id, parent):
        return c_user(
            uuid, [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}], parent = parent
        )

    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            block("a1", "t1", "u1"),
            {"type": "assistant", "uuid": "x", "parentUuid": "a1", "message": None},
            block("a1b", "t2", "x"),
            result("r2", "t2", "a1b"),
            result("r1", "t1", "a1"),
            c_asst("a2", [{"type": "text", "text": "A"}], parent = "r1"),
        ],
    )
    messages = claude.read_transcript(path, "t", "s1").messages
    ids = [m["id"] for m in messages]
    assert len(messages) == 4
    assert [m["parentId"] for m in messages] == [None, *ids[:-1]]


def test_cursor_reads_the_query_and_strips_injected_context(cursor_home):
    path = _write(
        cursor_home / "projects" / "p" / "agent-transcripts" / "s1" / "s1.jsonl",
        [
            k_turn("user", "<attached_files>x</attached_files><user_query>Why slow?</user_query>"),
            {
                "role": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "[REDACTED]\nSource maps."},
                        {"type": "tool_use", "name": "Read", "input": {"path": "a"}},
                    ]
                },
            },
            {"type": "session_event"},
        ],
    )
    t = cursor.read_transcript(path, "t", "s1")

    assert t.messages[0]["content"] == [{"type": "text", "text": "Why slow?"}]
    assert t.messages[1]["content"][0] == {"type": "text", "text": "Source maps."}
    assert t.messages[1]["content"][1]["toolName"] == "Read"
    assert t.messages[1]["parentId"] == t.messages[0]["id"]


def test_cursor_keeps_pasted_images(cursor_home):
    project = cursor_home / "projects" / "p"
    image = project / "assets" / "image-1.png"
    image.parent.mkdir(parents = True)
    image.write_bytes(base64.b64decode(PNG))
    path = _write(
        project / "agent-transcripts" / "s1" / "s1.jsonl",
        [
            k_turn(
                "user",
                "[Image]\n<image_files>\nThe following images were provided by the user and saved"
                f" to the workspace for future use:\n1. {image}\n2. {project}/assets/gone.png\n"
                "</image_files>\n<user_query>\nWhy does this button look broken?\n</user_query>",
            ),
            k_turn("assistant", "The padding is off."),
        ],
    )
    t = cursor.read_transcript(path, "t", "s1")

    assert t.messages[0]["content"] == [
        {"type": "image", "image": f"data:image/png;base64,{PNG}"},
        {"type": "text", "text": "Why does this button look broken?"},
    ]
    assert t.title == "Why does this button look broken?"


def test_cursor_drops_the_placeholder_beside_an_image_only_paste(cursor_home):
    project = cursor_home / "projects" / "p"
    image = project / "assets" / "image-1.png"
    image.parent.mkdir(parents = True)
    image.write_bytes(base64.b64decode(PNG))
    pasted = f"[Image]\n<image_files>\n1. {image}\n</image_files>\n<user_query>\n</user_query>"
    path = _write(
        project / "agent-transcripts" / "s1" / "s1.jsonl",
        [
            k_turn("user", pasted),
            k_turn("assistant", "A button."),
            k_turn("user", pasted.replace(str(image), str(project / "assets" / "gone.png"))),
        ],
    )
    t = cursor.read_transcript(path, "t", "s1")

    assert t.messages[0]["content"] == [{"type": "image", "image": f"data:image/png;base64,{PNG}"}]
    # No image left to show: the turn keeps its text so later message ids do not shift.
    assert len(t.messages) == 3 and t.messages[2]["content"][0]["type"] == "text"


def test_codex_replays_events_not_the_injected_context(codex_home):
    t = codex.read_transcript(_rollout(codex_home), "t", "s1")

    assert [m["role"] for m in t.messages] == ["user", "assistant", "assistant", "assistant"]
    assert t.messages[0]["content"] == [{"type": "text", "text": "List the files"}]
    assert t.messages[1]["content"] == [
        {
            "type": "tool-call",
            "toolCallId": "c1",
            "toolName": "shell",
            "args": {"command": ["ls"]},
            "result": "a.py",
        }
    ]
    assert t.messages[2]["content"][0]["args"] == {"input": "*** Begin Patch"}
    assert t.messages[2]["content"][0]["result"] == "Done!"
    assert t.messages[3]["content"] == [{"type": "text", "text": "One file: a.py"}]
    assert t.title == "List the files"


def test_codex_groups_by_cwd_and_skips_subagents(codex_home, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/Users/me")))
    _rollout(codex_home, sid = "a")
    _rollout(codex_home, sid = "b")
    _rollout(codex_home, cwd = "/srv/api", sid = "c")
    _rollout(codex_home, sid = "d", source = {"subagent": "review"})

    projects = {p.name: sorted(s.name for s in p.sessions) for p in codex.list_projects(codex_home)}
    assert projects == {
        "app": ["rollout-2026-09-22T12-00-00-a.jsonl", "rollout-2026-09-22T12-00-00-b.jsonl"],
        "srv-api": ["rollout-2026-09-22T12-00-00-c.jsonl"],
    }


def test_codex_compressing_a_rollout_keeps_its_chat(codex_home):
    zstandard = pytest.importorskip("zstandard")
    path = _rollout(codex_home)
    assert run_import(codex.SOURCE).new_chats == 1
    zst = path.with_name(path.name + ".zst")
    zst.write_bytes(zstandard.ZstdCompressor().compress(path.read_bytes()))
    path.unlink()

    again = run_import(codex.SOURCE)
    assert (again.new_chats, again.messages, again.warnings) == (0, 0, [])


def test_codex_without_a_zstd_decoder_warns_instead_of_failing(codex_home, monkeypatch):
    import builtins

    path = _rollout(codex_home)
    path.rename(path.with_name(path.name + ".zst"))
    real_import = builtins.__import__

    def no_zstd(name, *args, **kwargs):
        if name in ("zstandard", "compression"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_zstd)
    summary = run_import(codex.SOURCE)
    assert summary.new_chats == 0
    assert summary.warnings and "zstandard" in summary.warnings[0]


# Discovery


def test_project_names_drop_the_home_folder(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/Users/me")))
    assert display_name("-Users-me-dev-my-app") == "dev-my-app"
    assert display_name("users-ME-x") == "x"
    assert display_name("-opt-tool") == "opt-tool"


def test_cursor_files_a_shared_session_under_its_folder(cursor_home):
    for slug in (cursor.NO_FOLDER_SLUG, "Users-me-app"):
        _write(
            cursor_home / "projects" / slug / "agent-transcripts" / "s1" / "s1.jsonl",
            [k_turn("user", "hi")],
        )
    _write(
        cursor_home / "projects" / cursor.NO_FOLDER_SLUG / "agent-transcripts" / "s2.jsonl",
        [k_turn("user", "yo")],
    )

    projects = {p.slug: [s.stem for s in p.sessions] for p in cursor.list_projects(cursor_home)}
    assert projects == {"Users-me-app": ["s1"], cursor.NO_FOLDER_SLUG: ["s2"]}


# Importing


def test_import_groups_by_project_and_a_second_run_is_a_no_op(claude_home):
    _session(claude_home)
    first = run_import(claude.SOURCE)
    second = run_import(claude.SOURCE)

    assert (first.projects, first.chats, first.new_chats, first.messages) == (1, 1, 1, 2)
    assert (second.new_chats, second.messages) == (0, 0)
    assert studio_db.get_chat_project(project_id_for(claude.SOURCE, "-Users-me-app"))[
        "name"
    ].startswith("Claude · ")


def test_reimport_appends_new_turns_and_keeps_studio_edits(claude_home):
    path = _session(claude_home)
    run_import(claude.SOURCE)
    tid = thread_id_for(claude.SOURCE, "s1")
    studio_db.update_chat_thread(tid, {"title": "Renamed", "projectId": None})
    first, second = _messages(claude.SOURCE, "s1")
    # Edit the prompt, delete the reply.
    studio_db.sync_chat_messages(
        tid, [{**first, "content": [{"type": "text", "text": "edited"}]}], prune_missing = True
    )
    _append(path, [c_user("u2", "More", parent = "a1", ts = "2026-08-01T10:00:02Z")])

    summary = run_import(claude.SOURCE)
    thread = studio_db.get_chat_thread(tid)
    rows = {m["id"]: m for m in _messages(claude.SOURCE, "s1")}

    assert summary.messages == 1
    assert (thread["title"], thread["projectId"]) == ("Renamed", None)
    assert rows[first["id"]]["content"][0]["text"] == "edited"
    assert second["id"] not in rows
    new = next(m for m in rows.values() if m["content"][0].get("text") == "More")
    assert new["parentId"] in rows


def test_a_chat_imported_before_commands_were_kept_resumes_after_its_turns(claude_home):
    command = (
        "<command-message>explain</command-message>\n<command-name>/explain</command-name>\n"
        "<command-args>{}</command-args>"
    )
    path = _session(
        claude_home,
        records = [
            c_user("u1", "First"),
            c_asst("a1", [{"type": "text", "text": "One."}], parent = "u1"),
            c_user("cmd", command.format("the loop"), parent = "a1"),
            c_user(
                "exp", [{"type": "text", "text": "Explain: the loop"}], parent = "cmd", isMeta = True
            ),
            c_asst("a2", [{"type": "text", "text": "Two."}], parent = "exp"),
        ],
    )
    run_import(claude.SOURCE)
    tid = thread_id_for(claude.SOURCE, "s1")
    rows = {m["content"][0]["text"]: m for m in _messages(claude.SOURCE, "s1")}
    # What the earlier parser left: no command turn, 3 turns under revision "".
    conn = studio_db.get_connection()
    conn.execute("DELETE FROM chat_messages WHERE id = ?", (rows["/explain the loop"]["id"],))
    conn.execute("UPDATE external_import_sessions SET turns_imported = 3, revision = ''")
    conn.commit()
    conn.close()
    edited = {**rows["Two."], "content": [{"type": "text", "text": "edited"}]}
    studio_db.sync_chat_messages(tid, [edited], prune_missing = False)
    _append(
        path,
        [
            c_user("cmd2", command.format("the cache"), parent = "a2", ts = "2026-08-01T10:00:02Z"),
            c_user("exp2", "Explain: the cache", parent = "cmd2", isMeta = True),
            c_asst("a3", [{"type": "text", "text": "Three."}], parent = "exp2"),
        ],
    )

    assert run_import(claude.SOURCE).messages == 2
    texts = {m["id"]: m["content"][0]["text"] for m in _messages(claude.SOURCE, "s1")}
    assert sorted(texts.values()) == ["/explain the cache", "First", "One.", "Three.", "edited"]
    assert run_import(claude.SOURCE).messages == 0


def test_a_deleted_chat_stays_deleted_until_studio_is_emptied(claude_home):
    _session(claude_home, sid = "s1")
    _session(claude_home, sid = "s2")
    run_import(claude.SOURCE)
    studio_db.delete_chat_threads([thread_id_for(claude.SOURCE, "s1")])

    assert run_import(claude.SOURCE).skipped == 1
    assert studio_db.get_chat_thread(thread_id_for(claude.SOURCE, "s1")) is None

    studio_db.delete_chat_threads([t["id"] for t in studio_db.list_chat_threads()])
    assert run_import(claude.SOURCE).new_chats == 2


def test_a_tool_result_that_arrives_later_reaches_the_stored_call(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("u1", "List"),
            c_asst(
                "a1", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], parent = "u1"
            ),
        ],
    )
    run_import(claude.SOURCE)
    _append(
        path,
        [
            c_user(
                "r1", [{"type": "tool_result", "tool_use_id": "t1", "content": "out"}], parent = "a1"
            )
        ],
    )
    run_import(claude.SOURCE)

    call = _messages(claude.SOURCE, "s1")[1]["content"][0]
    assert call["result"] == "out"


@pytest.mark.parametrize(
    "content, legacy",
    [([IMAGE_RESULT], ""), ([{"type": "text", "text": "Page 1"}, IMAGE_RESULT], "Page 1")],
)
def test_a_chat_imported_before_tool_images_were_kept_gets_them_on_reimport(
    claude_home, content, legacy
):
    _read_image(claude_home, content)
    run_import(claude.SOURCE)
    tid = thread_id_for(claude.SOURCE, "s1")
    _, call, answer = _messages(claude.SOURCE, "s1")
    emptied = {**call, "content": [{**call["content"][0], "result": legacy}]}
    edited = {**answer, "content": [{"type": "text", "text": "edited"}]}
    studio_db.sync_chat_messages(tid, [emptied, edited], prune_missing = False)

    assert run_import(claude.SOURCE).messages == 1
    rows = _messages(claude.SOURCE, "s1")
    assert rows[1]["content"][0]["result"]["images"] == [{"data": PNG, "mimeType": "image/png"}]
    assert rows[2]["content"] == [{"type": "text", "text": "edited"}]
    assert run_import(claude.SOURCE).messages == 0


def test_reimport_keeps_a_tool_result_that_differs_from_the_new_text(claude_home):
    _read_image(claude_home, [{"type": "text", "text": "Page 1"}, IMAGE_RESULT])
    run_import(claude.SOURCE)
    tid = thread_id_for(claude.SOURCE, "s1")
    call = _messages(claude.SOURCE, "s1")[1]
    kept = {**call, "content": [{**call["content"][0], "result": "Page 1 (edited)"}]}
    studio_db.sync_chat_messages(tid, [kept], prune_missing = False)

    run_import(claude.SOURCE)
    assert _messages(claude.SOURCE, "s1")[1]["content"][0]["result"] == "Page 1 (edited)"


def test_an_interrupted_import_is_retried(claude_home):
    _session(claude_home)
    transcript = claude.read_transcript(
        next((claude_home / "projects").rglob("*.jsonl")), thread_id_for(claude.SOURCE, "s1"), "s1"
    )
    studio_db.upsert_chat_thread(
        {
            "id": transcript.thread_id,
            "title": "x",
            "modelType": "base",
            "modelId": "",
            "createdAt": 1,
            "updatedAt": 1,
        }
    )

    s = run_import(claude.SOURCE)
    print("PENDING", s.messages, studio_db.get_external_import_mark("claude", "s1"))


# Routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix = "/api/import")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_routes_report_status_and_import(client, claude_home, cursor_home, codex_home):
    _session(claude_home)
    assert client.get("/api/import/claude/status").json() == {"available": True, "chats": 1}
    assert client.get("/api/import/cursor/status").json()["available"] is False
    assert client.get("/api/import/codex/status").json()["available"] is False
    body = client.post("/api/import/claude").json()
    assert (body["new_chats"], body["messages"], body["warnings"]) == (1, 2, [])
    assert client.post("/api/import/vscode").status_code == 422


def test_routes_need_a_signed_in_user(claude_home):
    app = FastAPI()
    app.include_router(router, prefix = "/api/import")
    assert TestClient(app).post("/api/import/claude").status_code in (401, 403)


@pytest.mark.parametrize("source", ["claude", "cursor", "codex"])
def test_a_managed_account_cannot_reach_the_owners_histories(
    claude_home, cursor_home, codex_home, source
):
    from utils.account_context import AccountContext, bind_account, reset_account

    _session(claude_home)
    _write(
        cursor_home / "projects" / "p" / "agent-transcripts" / "s1.jsonl", [k_turn("user", "hi")]
    )
    app = FastAPI()
    app.include_router(router, prefix = "/api/import")
    app.dependency_overrides[get_current_subject] = lambda: "alice"

    @app.middleware("http")
    async def as_managed_account(request, call_next):
        token = bind_account(AccountContext("acct-alice", "alice"))
        try:
            return await call_next(request)
        finally:
            reset_account(token)

    managed = TestClient(app)
    assert managed.get(f"/api/import/{source}/status").json()["available"] is False
    assert managed.post(f"/api/import/{source}").status_code == 403
    assert studio_db.list_chat_threads() == []


def test_every_appended_branch_is_reparented_past_a_deleted_message(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("u1", "one"),
            c_asst("a1", [{"type": "text", "text": "A"}], parent = "u1"),
        ],
    )
    run_import(claude.SOURCE)
    tid = thread_id_for(claude.SOURCE, "s1")
    studio_db.sync_chat_messages(tid, _messages(claude.SOURCE, "s1")[:1], prune_missing = True)
    # Two rewinds off the deleted reply.
    _append(path, [c_user("u2", "two", parent = "a1"), c_user("u3", "three", parent = "a1")])
    run_import(claude.SOURCE)

    rows = _messages(claude.SOURCE, "s1")
    ids = {m["id"] for m in rows}
    assert len(rows) == 3
    assert all(m["parentId"] is None or m["parentId"] in ids for m in rows)


def test_an_emptied_studio_keeps_unrelated_tombstones(claude_home):
    _session(claude_home)
    studio_db.upsert_chat_thread(
        {
            "id": "native",
            "title": "mine",
            "modelType": "base",
            "modelId": "",
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    studio_db.delete_chat_threads(["native"])

    assert run_import(claude.SOURCE).new_chats == 1
    with pytest.raises(studio_db.ChatThreadDeletedError):
        studio_db.upsert_chat_thread(
            {
                "id": "native",
                "title": "stale tab",
                "modelType": "base",
                "modelId": "",
                "createdAt": 1,
                "updatedAt": 2,
            }
        )


def test_codex_cwds_that_only_differ_in_punctuation_stay_apart(codex_home):
    _rollout(codex_home, cwd = "/work/a-b", sid = "a")
    _rollout(codex_home, cwd = "/work/a/b", sid = "b")
    assert len(codex.list_projects(codex_home)) == 2


def test_codex_status_counts_rollouts_without_opening_them(codex_home, monkeypatch):
    _rollout(codex_home, sid = "a")
    _rollout(codex_home, sid = "b")
    monkeypatch.setattr(codex, "_meta", lambda path: pytest.fail("status must not parse rollouts"))
    assert codex.SOURCE.session_count() == 2


def test_codex_reads_archived_rollouts_once(codex_home):
    _rollout(codex_home, sid = "a")
    _rollout(codex_home, sid = "b", root = "archived_sessions")
    _rollout(codex_home, sid = "a", root = "archived_sessions")

    sessions = sorted(s.name for p in codex.list_projects(codex_home) for s in p.sessions)
    assert sessions == [
        "rollout-2026-09-22T12-00-00-a.jsonl",
        "rollout-2026-09-22T12-00-00-b.jsonl",
    ]


def _continuation(home, name, thread, base_thread, cut, text):
    meta = {
        "id": thread,
        "cwd": "/Users/me/app",
        "source": "cli",
        "history_base": {"thread_id": base_thread, "end_ordinal_exclusive": cut},
    }
    path = home / "sessions" / "2026" / "09" / "23" / f"rollout-2026-09-23T12-00-00-{name}.jsonl"
    return _write(
        path,
        [
            {**x_line("session_meta", meta), "ordinal": cut},
            {**x_line("event_msg", {"type": "user_message", "message": text}), "ordinal": cut + 1},
        ],
    )


def test_codex_revert_and_fork_rebuild_the_inherited_prefix(codex_home):
    base = _rollout(codex_home, sid = "t1")
    lines = [json.loads(line) for line in base.read_text().splitlines()]
    base.write_text("".join(json.dumps({**r, "ordinal": i}) + "\n" for i, r in enumerate(lines)))
    assert run_import(codex.SOURCE).new_chats == 1
    tid = thread_id_for(codex.SOURCE, "t1")
    before = [m["id"] for m in _messages(codex.SOURCE, "t1")]

    # Revert to just after the first prompt (ordinal 4), then a new prompt; plus a fork there.
    _continuation(codex_home, "t1_r2", "t1", "t1", 4, "Try again")
    _continuation(codex_home, "f1", "f1", "t1", 4, "Forked question")
    summary = run_import(codex.SOURCE)

    rows = {m["id"]: m for m in studio_db.list_chat_messages(tid)}
    retry = next(m for m in rows.values() if m["content"][0].get("text") == "Try again")
    assert set(before) <= set(rows)  # the reverted turns stay, as a sibling branch
    assert rows[retry["parentId"]]["content"][0]["text"] == "List the files"
    assert summary.new_chats == 1  # the fork, not a second copy of the reverted thread
    fork = [m["content"][0].get("text") for m in _messages(codex.SOURCE, "f1")]
    assert fork == ["List the files", "Forked question"]

    # A fork of the revert names the replacement rollout ("r2"), not the thread id.
    _continuation(codex_home, "f2", "f2", "r2", 6, "Fork of the retry")
    run_import(codex.SOURCE)
    rows = {m["id"]: m for m in _messages(codex.SOURCE, "f2")}
    tip = next(m for m in rows.values() if m["content"][0].get("text") == "Fork of the retry")
    chain = []
    while tip:
        chain.append(tip["content"][0].get("text"))
        tip = rows.get(tip["parentId"])
    assert chain == ["Fork of the retry", "Try again", "List the files"]


_PAGINATED_ROLLOUT = Path(__file__).parent / "fixtures" / "codex_paginated_rollout.jsonl"
_PAGINATED_SID = "01a1d5af-1734-7651-b549-42d9e5803184"


def _paginated_rollout(home, old = False):
    """``old``: what the reader before completed items saw, so the import stores an untagged mark."""
    records = [
        json.loads(line) for line in _PAGINATED_ROLLOUT.read_text(encoding = "utf-8").splitlines()
    ]
    if old:
        records = [r for r in records if r["payload"].get("type") != "item_completed"]
        records[0]["payload"].pop("history_mode")
    name = f"rollout-2026-11-13T16-13-04-{_PAGINATED_SID}.jsonl"
    return _write(home / "sessions" / "2026" / "11" / "13" / name, records)


def _turns(sid):
    return [
        (m["role"], m["content"][0].get("text") or m["content"][0]["toolName"])
        for m in _messages(codex.SOURCE, sid)
    ]


_PAGINATED_TURNS = [
    ("user", "Why does checkout fail on empty carts? MARK-N1"),
    ("assistant", "Codex reply to MARK-N1"),
    ("user", "[RUN] run the cart tests MARK-N2"),
    ("assistant", "shell"),
    ("assistant", "Codex reply to MARK-N2 after tool"),
    ("user", "Great, now summarise the fix MARK-N3"),
    ("assistant", "Codex reply to MARK-N3"),
]


def test_codex_imports_every_turn_of_a_current_rollout(codex_home):
    _paginated_rollout(codex_home)
    summary = run_import(codex.SOURCE)

    assert (summary.new_chats, summary.messages) == (1, 7)
    assert _turns(_PAGINATED_SID) == _PAGINATED_TURNS
    thread = studio_db.get_chat_thread(thread_id_for(codex.SOURCE, _PAGINATED_SID))
    assert thread["title"] == "Why does checkout fail on empty carts? MARK-N1"


def _completed(
    kind,
    text,
    ts = "2026-11-13T16:20:00Z",
):
    part = "text" if kind == "UserMessage" else "Text"
    item = {"type": kind, "id": text, "content": [{"type": part, "text": text}]}
    return x_line("event_msg", {"type": "item_completed", "item": item}, ts = ts)


def test_codex_reimport_brings_turns_an_older_import_missed(codex_home):
    path = _paginated_rollout(codex_home, old = True)
    assert run_import(codex.SOURCE).messages == 1
    _paginated_rollout(codex_home)

    assert run_import(codex.SOURCE).messages == 7
    assert _turns(_PAGINATED_SID) == _PAGINATED_TURNS
    assert run_import(codex.SOURCE).messages == 0

    _append(path, [{**_completed("UserMessage", "One more"), "ordinal": 38}])
    assert run_import(codex.SOURCE).messages == 1
    assert _turns(_PAGINATED_SID) == [*_PAGINATED_TURNS, ("user", "One more")]


def _old_import(home):
    _paginated_rollout(home, old = True)
    run_import(codex.SOURCE)
    _paginated_rollout(home)
    tid = thread_id_for(codex.SOURCE, _PAGINATED_SID)
    return tid, _messages(codex.SOURCE, _PAGINATED_SID)[0]


def _parent_texts(sid):
    rows = {m["id"]: m for m in _messages(codex.SOURCE, sid)}

    def label(m):
        return m["content"][0].get("text") or m["content"][0]["toolName"]

    return {label(m): label(rows[m["parentId"]]) if m["parentId"] else None for m in rows.values()}


def test_codex_backfill_keeps_an_edited_turn_and_moves_it_under_its_prompt(codex_home):
    tid, shell = _old_import(codex_home)
    edited = [{**shell["content"][0], "result": "edited in Studio"}]
    studio_db.sync_chat_messages(tid, [{**shell, "content": edited}], prune_missing = True)

    run_import(codex.SOURCE)
    rows = {m["id"]: m for m in _messages(codex.SOURCE, _PAGINATED_SID)}
    assert rows[shell["id"]]["content"] == edited
    assert _parent_texts(_PAGINATED_SID)["shell"] == "[RUN] run the cart tests MARK-N2"
    assert _turns(_PAGINATED_SID) == _PAGINATED_TURNS


def test_codex_backfill_leaves_a_deleted_turn_deleted(codex_home):
    tid, shell = _old_import(codex_home)
    studio_db.sync_chat_messages(tid, [], prune_missing = True)

    assert run_import(codex.SOURCE).messages == 6
    assert _turns(_PAGINATED_SID) == [t for t in _PAGINATED_TURNS if t[1] != "shell"]
    parents = _parent_texts(_PAGINATED_SID)
    assert parents["Codex reply to MARK-N2 after tool"] == "[RUN] run the cart tests MARK-N2"


def test_an_empty_tool_output_still_completes_the_call(claude_home):
    path = _session(
        claude_home,
        records = [
            c_user("u1", "Touch it"),
            c_asst(
                "a1", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], parent = "u1"
            ),
            c_user(
                "r1", [{"type": "tool_result", "tool_use_id": "t1", "content": ""}], parent = "a1"
            ),
        ],
    )
    call = claude.read_transcript(path, "t", "s1").messages[1]["content"][0]
    assert call["result"] == ""


def test_codex_revert_after_an_older_import_brings_the_missed_prefix(codex_home):
    _paginated_rollout(codex_home, old = True)
    assert run_import(codex.SOURCE).messages == 1
    _paginated_rollout(codex_home)
    meta = {
        "id": _PAGINATED_SID,
        "cwd": "/Users/me/webshop",
        "history_base": {"thread_id": _PAGINATED_SID, "end_ordinal_exclusive": 22},
    }
    name = f"rollout-2026-11-14T10-00-00-{_PAGINATED_SID}_r2.jsonl"
    _write(
        codex_home / "sessions" / "2026" / "11" / "14" / name,
        [
            {**x_line("session_meta", meta), "ordinal": 22},
            {**_completed("UserMessage", "Try again"), "ordinal": 23},
        ],
    )
    run_import(codex.SOURCE)
    assert _turns(_PAGINATED_SID) == [*_PAGINATED_TURNS[:4], ("user", "Try again")]


def _pasted(text, image = True):
    request = [
        {"type": "input_text", "text": '<image name=[Image #1] path="shot.png">'},
        {"type": "input_image", "image_url": f"data:image/png;base64,{PNG}", "detail": "high"},
        {"type": "input_text", "text": "</image>"},
        {"type": "input_text", "text": text},
    ]
    content = [{"type": "local_image", "path": "shot.png"}, {"type": "text", "text": text}]
    if not image:
        request, content = request[3:], content[1:]
    item = {"type": "UserMessage", "id": text, "content": content}
    return [
        x_line("response_item", {"type": "message", "role": "user", "content": request}),
        x_line("event_msg", {"type": "item_completed", "item": item}),
    ]


def _pasted_rollout(home, image = True):
    body = [
        *_pasted("What colour is this?", image),
        _completed("AgentMessage", "Red"),
        *_pasted("", image),
        _completed("AgentMessage", "Blue"),
    ]
    return _rollout(home, sid = "img", history_mode = "paginated", body = body)


def test_codex_keeps_pasted_and_attached_images(codex_home):
    image = {"type": "image", "image": f"data:image/png;base64,{PNG}"}
    messages = codex.read_transcript(_pasted_rollout(codex_home), "t", "s1").messages

    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", [image, {"type": "text", "text": "What colour is this?"}]),
        ("assistant", [{"type": "text", "text": "Red"}]),
        ("user", [image]),
        ("assistant", [{"type": "text", "text": "Blue"}]),
    ]

    legacy = _rollout(
        codex_home,
        sid = "legacy",
        body = [
            _pasted("What colour is this?")[0],
            x_line(
                "event_msg",
                {
                    "type": "user_message",
                    "message": "What colour is this?",
                    "images": [],
                    "local_images": ["shot.png"],
                },
            ),
            x_line("event_msg", {"type": "agent_message", "message": "Red"}),
            x_line("event_msg", {"type": "user_message", "message": "And now?"}),
        ],
    )
    messages = codex.read_transcript(legacy, "t", "s2").messages
    assert [m["content"] for m in messages] == [
        [image, {"type": "text", "text": "What colour is this?"}],
        [{"type": "text", "text": "Red"}],
        [{"type": "text", "text": "And now?"}],
    ]


def test_codex_reimport_after_images_keeps_studio_edits(codex_home):
    _pasted_rollout(codex_home, image = False)
    assert run_import(codex.SOURCE).messages == 3
    tid = thread_id_for(codex.SOURCE, "img")
    reply = next(m for m in _messages(codex.SOURCE, "img") if m["content"][0]["text"] == "Blue")
    edited = [{"type": "text", "text": "Blue, edited in Studio"}]
    studio_db.sync_chat_messages(tid, [{**reply, "content": edited}], prune_missing = False)

    path = _pasted_rollout(codex_home)
    assert run_import(codex.SOURCE).messages == 0
    assert {m["id"]: m for m in _messages(codex.SOURCE, "img")}[reply["id"]]["content"] == edited

    _append(path, [_completed("UserMessage", "One more")])
    assert run_import(codex.SOURCE).messages == 1


def _tool_images_rollout(
    home,
    generated = True,
    pasted = True,
):
    def call(call_id, name, output):
        return [
            x_line(
                "response_item",
                {"type": "function_call", "name": name, "arguments": "{}", "call_id": call_id},
            ),
            x_line(
                "response_item",
                {"type": "function_call_output", "call_id": call_id, "output": output},
            ),
        ]

    body = [
        *_pasted("", pasted),
        _completed("AgentMessage", "Taking a look."),
        *call(
            "c1",
            "view_image",
            [{"type": "input_image", "image_url": f"data:application/octet-stream;base64,{PNG}"}],
        ),
        *call(
            "c2",
            "mcp__playwright__browser_take_screenshot",
            [
                {"type": "input_text", "text": "Took the screenshot"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{JPEG}"},
            ],
        ),
        _completed("AgentMessage", "A login form. Here is an icon for it."),
        x_line(
            "response_item",
            {
                "type": "image_generation_call",
                "id": "ig_1",
                "status": "completed",
                "revised_prompt": "A flat login icon",
                "result": PNG,
            },
        ),
        _completed("AgentMessage", "Done."),
    ]
    if not generated:
        # A skipped record, so later records keep their ordinals and message ids.
        body = [
            x_line("response_item", {"type": "reasoning"})
            if line["payload"]["type"] == "image_generation_call"
            else line
            for line in body
        ]
    return _rollout(home, sid = "tools", history_mode = "paginated", body = body)


def test_codex_keeps_images_tools_viewed_and_generated(codex_home):
    shot = {"data": PNG, "mimeType": "image/png"}
    messages = codex.read_transcript(_tool_images_rollout(codex_home), "t", "s1").messages
    calls = [m["content"][0] for m in messages if m["content"][0]["type"] == "tool-call"]

    assert [(c["toolName"], c["result"]) for c in calls] == [
        ("view_image", {"text": "", "images": [shot]}),
        (
            "mcp__playwright__browser_take_screenshot",
            {"text": "Took the screenshot", "images": [{"data": JPEG, "mimeType": "image/jpeg"}]},
        ),
        ("image_generation", {"image_b64": PNG, "image_mime": "image/png"}),
    ]
    assert calls[2]["args"] == {"prompt": "A flat login icon", "_server_tool": True}
    assert messages[-1]["content"] == [{"type": "text", "text": "Done."}]


# Without the pasted image the first import is marked ":items", with it ":images".
@pytest.mark.parametrize("pasted", [True, False])
def test_codex_reimport_after_generated_images_keeps_studio_edits(codex_home, pasted):
    _tool_images_rollout(codex_home, generated = False, pasted = pasted)
    assert run_import(codex.SOURCE).messages == 5 + pasted
    tid = thread_id_for(codex.SOURCE, "tools")
    reply = next(
        m for m in _messages(codex.SOURCE, "tools") if m["content"][0].get("text") == "Done."
    )
    edited = [{"type": "text", "text": "Done, edited in Studio"}]
    # Results as stored before tool images were kept: text only.
    legacy = [
        {**m, "content": [{**m["content"][0], "result": m["content"][0]["result"]["text"]}]}
        for m in _messages(codex.SOURCE, "tools")
        if isinstance(m["content"][0].get("result"), dict)
    ]
    studio_db.sync_chat_messages(tid, [{**reply, "content": edited}, *legacy], prune_missing = False)

    path = _tool_images_rollout(codex_home, pasted = pasted)
    _append(path, [_completed("UserMessage", "Continued in Codex")])
    assert run_import(codex.SOURCE).messages == 3
    assert "Continued in Codex" in _texts("tools")
    stored = {m["id"]: m for m in _messages(codex.SOURCE, "tools")}
    restored = {
        stored[m["id"]]["content"][0]["toolName"]: stored[m["id"]]["content"][0]["result"]
        for m in legacy
    }
    assert restored == {
        "view_image": {"text": "", "images": [{"data": PNG, "mimeType": "image/png"}]},
        "mcp__playwright__browser_take_screenshot": {
            "text": "Took the screenshot",
            "images": [{"data": JPEG, "mimeType": "image/jpeg"}],
        },
    }
    assert stored[reply["id"]]["content"] == edited

    _append(path, [_completed("UserMessage", "One more")])
    assert run_import(codex.SOURCE).messages == 1


def _image_revert(
    home,
    cut,
    tail,
    image = True,
):
    body = [
        _completed("UserMessage", "Hello"),
        _completed("AgentMessage", "Hi"),
        *_pasted("", image),
        _completed("AgentMessage", "Blue"),
    ]
    base = _rollout(home, sid = "img", history_mode = "paginated", body = body)
    meta = {
        "id": "img",
        "cwd": "/Users/me/app",
        "history_mode": "paginated",
        "history_base": {"thread_id": "img", "end_ordinal_exclusive": cut},
    }
    records = [x_line("session_meta", meta), *tail]
    name = "rollout-2026-09-22T12-00-01-img_r2.jsonl"
    revert = home / "sessions" / "2026" / "09" / "22" / name
    return base, lambda: _write(revert, [{**r, "ordinal": cut + i} for i, r in enumerate(records)])


def _texts(sid):
    return [m["content"][-1].get("text") for m in _messages(codex.SOURCE, sid)]


@pytest.mark.parametrize(
    "image, tail",
    [
        (True, [_completed("UserMessage", "Try again")]),
        (False, [*_pasted("")]),
        (
            False,
            [x_line("response_item", {"type": "image_generation_call", "id": "ig", "result": PNG})],
        ),
    ],
)
def test_codex_revert_across_image_turns_leaves_a_deleted_turn_deleted(codex_home, image, tail):
    _, revert = _image_revert(codex_home, 5, tail, image)
    run_import(codex.SOURCE)
    tid = thread_id_for(codex.SOURCE, "img")
    kept = [m for m in _messages(codex.SOURCE, "img") if m["content"][-1].get("text") != "Hi"]
    studio_db.sync_chat_messages(tid, kept, prune_missing = True)

    revert()
    assert run_import(codex.SOURCE).messages == 1
    assert "Hi" not in _texts("img")


def test_codex_deleting_an_image_base_keeps_the_revert_tail_edits(codex_home):
    base, revert = _image_revert(codex_home, 7, [_completed("UserMessage", "Try again")])
    revert()
    assert run_import(codex.SOURCE).messages == 4
    tid = thread_id_for(codex.SOURCE, "img")
    tip = next(
        m for m in _messages(codex.SOURCE, "img") if m["content"][-1].get("text") == "Try again"
    )
    edited = [{"type": "text", "text": "Edited in Studio"}]
    studio_db.sync_chat_messages(tid, [{**tip, "content": edited}], prune_missing = False)

    base.unlink()
    assert run_import(codex.SOURCE).messages == 0
    assert "Edited in Studio" in _texts("img")


def test_codex_deleting_the_base_rollout_keeps_studio_edits(codex_home):
    base = _paginated_rollout(codex_home)
    meta = {
        "id": _PAGINATED_SID,
        "cwd": "/Users/me/webshop",
        "history_mode": "paginated",
        "history_base": {"thread_id": _PAGINATED_SID, "end_ordinal_exclusive": 38},
    }
    call = {"type": "function_call", "name": "shell", "arguments": "{}", "call_id": "c2"}
    name = f"rollout-2026-11-14T10-00-00-{_PAGINATED_SID}_r2.jsonl"
    _write(
        codex_home / "sessions" / "2026" / "11" / "14" / name,
        [
            {**x_line("session_meta", meta), "ordinal": 38},
            {**x_line("response_item", call, ts = "2026-11-14T10:00:01Z"), "ordinal": 39},
        ],
    )
    run_import(codex.SOURCE)
    tid = thread_id_for(codex.SOURCE, _PAGINATED_SID)
    rows = _messages(codex.SOURCE, _PAGINATED_SID)
    tip = next(m for m in rows if m["content"][0].get("toolCallId") == "c2")
    edited = [{**tip["content"][0], "result": "edited in Studio"}]
    studio_db.sync_chat_messages(tid, [{**tip, "content": edited}], prune_missing = False)

    base.unlink()
    run_import(codex.SOURCE)
    after = {m["id"]: m for m in _messages(codex.SOURCE, _PAGINATED_SID)}
    assert after[tip["id"]]["content"] == edited
    assert len(after) == len(rows)


def test_a_null_message_record_costs_only_itself(claude_home):
    _session(
        claude_home,
        records = [
            {"type": "user", "uuid": "b1", "message": None},
            c_user("u1", "Fix the header"),
            c_asst("a1", [{"type": "text", "text": "Fixed it."}], parent = "u1"),
        ],
    )

    summary = run_import(claude.SOURCE)

    assert summary.new_chats == 1 and summary.warnings == []
    assert [m["content"][0]["text"] for m in _messages(claude.SOURCE, "s1")] == [
        "Fix the header",
        "Fixed it.",
    ]


def test_a_session_that_crashes_costs_only_its_own_chat(claude_home):
    _session(claude_home, sid = "bad")
    _session(claude_home, sid = "good")

    def read(path, *args):
        if path.stem == "bad":
            raise ValueError("torn frame")
        return claude.read_transcript(path, *args)

    source = dataclasses.replace(claude.SOURCE, read_transcript = read)
    summary = run_import(source)

    assert summary.new_chats == 1
    assert [m["content"][0]["text"] for m in _messages(source, "good")] == [
        "Fix the header",
        "Fixed it.",
    ]
    assert summary.warnings == ["bad.jsonl: could not be imported (torn frame)."]
    assert run_import(source).new_chats == 0


def test_a_cyclic_parent_chain_reimports_instead_of_looping(claude_home):
    path = _session(claude_home)
    run_import(claude.SOURCE)
    transcript = claude.read_transcript(path, thread_id_for(claude.SOURCE, "s1"), "s1")
    cyclic = dataclasses.replace(
        transcript,
        messages = transcript.messages
        + [
            {**transcript.messages[0], "id": "cycle-x", "parentId": "cycle-y"},
            {**transcript.messages[1], "id": "cycle-y", "parentId": "cycle-x"},
        ],
    )
    source = dataclasses.replace(claude.SOURCE, read_transcript = lambda *args: cyclic)

    summary = run_import(source)

    rows = {m["id"]: m for m in _messages(source, "s1")}
    assert summary.messages == 2
    assert {"cycle-x", "cycle-y"} <= set(rows)
    assert all(m["parentId"] is None or m["parentId"] in rows for m in rows.values())
