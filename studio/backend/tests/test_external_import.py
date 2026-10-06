# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import storage.studio_db as studio_db
from auth.authentication import get_current_subject
from core.external_import import (
    claude,
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

    assert run_import(claude.SOURCE).messages == 2


# Routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix = "/api/import")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_routes_report_status_and_import(client, claude_home, cursor_home):
    _session(claude_home)
    assert client.get("/api/import/claude/status").json() == {
        "available": True,
        "projects": 1,
        "chats": 1,
    }
    assert client.get("/api/import/cursor/status").json()["available"] is False
    body = client.post("/api/import/claude").json()
    assert (body["new_chats"], body["messages"], body["warnings"]) == (1, 2, [])
    assert client.post("/api/import/vscode").status_code == 422


def test_routes_need_a_signed_in_user(claude_home):
    app = FastAPI()
    app.include_router(router, prefix = "/api/import")
    assert TestClient(app).post("/api/import/claude").status_code in (401, 403)


@pytest.mark.parametrize("source", ["claude", "cursor"])
def test_a_managed_account_cannot_reach_the_owners_histories(claude_home, cursor_home, source):
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
