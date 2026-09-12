# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Importing Claude Code's agent conversations into Studio."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import storage.studio_db as studio_db
from auth.authentication import get_current_subject
from core.claude_import import discovery, import_claude_chats, project_id_for, thread_id_for
from core.claude_import.transcripts import read_transcript
from routes.claude_import import router as claude_import_router

# ISO timestamps, the shape Claude Code writes.
T0 = "2026-08-01T10:00:00.000Z"
T1 = "2026-08-01T10:00:01.000Z"
T2 = "2026-08-01T10:00:02.000Z"
T3 = "2026-08-01T10:00:03.000Z"
T4 = "2026-08-01T10:00:04.000Z"
T5 = "2026-08-01T10:00:05.000Z"
T6 = "2026-08-01T10:00:06.000Z"


def user_record(
    uuid,
    text,
    timestamp,
    parent = None,
    **extra,
):
    record = {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": timestamp,
        "isSidechain": False,
        "message": {"role": "user", "content": text},
    }
    record.update(extra)
    return record


def assistant_record(
    uuid,
    content,
    timestamp,
    parent = None,
    **extra,
):
    record = {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": timestamp,
        "isSidechain": False,
        "message": {"role": "assistant", "content": content},
    }
    record.update(extra)
    return record


def tool_result_record(
    uuid,
    tool_use_id,
    result,
    timestamp,
    parent = None,
):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": "s1",
        "timestamp": timestamp,
        "isSidechain": False,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result}],
        },
    }


def text_block(text):
    return {"type": "text", "text": text}


def write_session(home, slug, session_id, records):
    """Lay out one session the way Claude Code does, and return its file."""
    project_dir = home / "projects" / slug
    project_dir.mkdir(parents = True, exist_ok = True)
    path = project_dir / (session_id + ".jsonl")
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding = "utf-8",
    )
    return path


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    """A Claude Code state directory with one project and one conversation."""
    home = tmp_path / "claude"
    write_session(
        home,
        "-Users-me-app",
        "session-one",
        [
            user_record("u1", "Fix the header", T0),
            assistant_record("a1", [text_block("Fixed it.")], T1, parent = "u1"),
        ],
    )
    monkeypatch.setenv(discovery.CLAUDE_HOME_ENV, str(home))
    return home


# Discovery


def test_lists_a_project_with_its_sessions(claude_home):
    projects = discovery.list_claude_projects()
    assert [(p.slug, len(p.sessions)) for p in projects] == [("-Users-me-app", 1)]


def test_skips_a_project_directory_holding_no_session(claude_home):
    (claude_home / "projects" / "-Users-me-empty").mkdir(parents = True)
    assert [p.slug for p in discovery.list_claude_projects()] == ["-Users-me-app"]


def test_skips_claude_own_bookkeeping_directories(claude_home):
    write_session(claude_home, ".internal", "session-two", [user_record("u1", "hi", T0)])
    assert [p.slug for p in discovery.list_claude_projects()] == ["-Users-me-app"]


def test_names_a_project_from_its_encoded_path(claude_home):
    assert discovery.list_claude_projects()[0].name == "Users/me/app"


# Transcript parsing


def test_keeps_real_timestamps_and_stable_ids(claude_home):
    path = claude_home / "projects" / "-Users-me-app" / "session-one.jsonl"
    transcript = read_transcript(path, "thread-1")
    assert [m["createdAt"] for m in transcript.messages] == [1785578400000, 1785578401000]
    again = read_transcript(path, "thread-1")
    assert [m["id"] for m in again.messages] == [m["id"] for m in transcript.messages]


def test_imports_a_tool_call_with_its_result(tmp_path):
    path = write_session(
        tmp_path,
        "-slug",
        "s1",
        [
            user_record("u1", "read it", T0),
            assistant_record(
                "a1",
                [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.txt"}}],
                T1,
                parent = "u1",
            ),
            tool_result_record("u2", "toolu_1", "file body", T2, parent = "a1"),
            assistant_record("a2", [text_block("Done.")], T3, parent = "u2"),
        ],
    )
    transcript = read_transcript(path, "thread-1")
    call = transcript.messages[1]["content"][0]
    assert transcript.tool_calls == 1
    assert call["type"] == "tool-call"
    assert call["toolName"] == "Read"
    assert call["args"] == {"path": "a.txt"}
    assert call["result"] == "file body"
    assert [m["role"] for m in transcript.messages] == ["user", "assistant", "assistant"]


def test_drops_thinking_but_keeps_the_prose(tmp_path):
    path = write_session(
        tmp_path,
        "-slug",
        "s1",
        [
            user_record("u1", "hi", T0),
            assistant_record(
                "a1",
                [{"type": "thinking", "thinking": "let me think"}, text_block("Hello.")],
                T1,
                parent = "u1",
            ),
        ],
    )
    transcript = read_transcript(path, "thread-1")
    assert transcript.messages[1]["content"] == [{"type": "text", "text": "Hello."}]


def test_skips_command_invocations_and_sidechains(tmp_path):
    path = write_session(
        tmp_path,
        "-slug",
        "s1",
        [
            user_record("u1", "<command-name>/exit</command-name>", T0),
            user_record("u2", "real question", T1, parent = "u1"),
            assistant_record("a1", [text_block("answer")], T2, parent = "u2"),
            user_record("s1", "subagent work", T2, parent = None, isSidechain = True),
        ],
    )
    transcript = read_transcript(path, "thread-1")
    assert [(m["role"], m["content"][0].get("text")) for m in transcript.messages] == [
        ("user", "real question"),
        ("assistant", "answer"),
    ]


def test_skips_the_meta_preamble_of_a_resumed_session(tmp_path):
    path = write_session(
        tmp_path,
        "-slug",
        "s1",
        [
            user_record(
                "u0",
                "Caveat: The messages below were generated by the user while running local commands.",
                T0,
                isMeta = True,
            ),
            user_record("u1", "real question", T1, parent = "u0"),
        ],
    )
    transcript = read_transcript(path, "thread-1")
    assert [m["content"][0]["text"] for m in transcript.messages] == ["real question"]


def test_cuts_command_output_appended_to_a_prompt(tmp_path):
    path = write_session(
        tmp_path,
        "-slug",
        "s1",
        [
            user_record(
                "u1",
                "Fix the header\n<local-command-stdout>Catch you later!</local-command-stdout>",
                T0,
            ),
        ],
    )
    transcript = read_transcript(path, "thread-1")
    assert [m["content"][0]["text"] for m in transcript.messages] == ["Fix the header"]


def test_passes_over_bookkeeping_and_a_half_written_line(tmp_path):
    path = write_session(
        tmp_path,
        "-slug",
        "s1",
        [
            user_record("u1", "hi", T0),
            {"type": "file-history-snapshot", "messageId": "u1"},
            {"type": "system", "subtype": "compact_boundary"},
        ],
    )
    with path.open("a", encoding = "utf-8") as handle:
        handle.write('{"type": "assis')
    transcript = read_transcript(path, "thread-1")
    assert [m["content"][0]["text"] for m in transcript.messages] == ["hi"]
    assert transcript.skipped_records >= 1


def test_titles_a_conversation_from_its_first_prompt(tmp_path):
    path = write_session(
        tmp_path, "-slug", "s1", [user_record("u1", "Fix the header\nand the footer", T0)]
    )
    assert read_transcript(path, "thread-1").title == "Fix the header"


def test_a_rewind_keeps_both_branches_on_the_same_parent(tmp_path):
    path = write_session(
        tmp_path,
        "-slug",
        "s1",
        [
            user_record("u1", "try this", T0),
            assistant_record("a1", [text_block("first")], T1, parent = "u1"),
            user_record("u2", "no, other way", T2, parent = "a1"),
            assistant_record("a2", [text_block("abandoned")], T3, parent = "u2"),
            user_record("u3", "retry", T4, parent = "a1"),
            assistant_record("a3", [text_block("kept")], T5, parent = "u3"),
        ],
    )
    transcript = read_transcript(path, "thread-1")
    assert [m["content"][0]["text"] for m in transcript.messages] == [
        "try this",
        "first",
        "no, other way",
        "abandoned",
        "retry",
        "kept",
    ]
    ids = [m["id"] for m in transcript.messages]
    parents = [m["parentId"] for m in transcript.messages]
    assert parents[0] is None
    assert parents[2] == ids[1]
    assert parents[4] == ids[1]


def test_a_long_parent_chain_does_not_overflow(tmp_path):
    records = [user_record("u0", "start", T0)]
    parent = "u0"
    for index in range(1, 1200):
        uuid = f"n{index}"
        if index % 2:
            records.append(assistant_record(uuid, [text_block(f"a{index}")], T1, parent = parent))
        else:
            records.append(user_record(uuid, f"u{index}", T1, parent = parent))
        parent = uuid
    path = write_session(tmp_path, "-slug", "s1", records)
    transcript = read_transcript(path, "thread-1")
    assert len(transcript.messages) == 1200


# Importing


def _threads_of(project_id):
    return [t for t in studio_db.list_chat_threads() if t["projectId"] == project_id]


def test_imports_one_project_per_claude_project(claude_home):
    summary = import_claude_chats()
    assert summary.projects == 1
    assert summary.chats == 1
    assert summary.new_chats == 1
    project = studio_db.list_chat_projects()[0]
    assert project["name"] == "Claude · Users/me/app"
    thread = _threads_of(project["id"])[0]
    assert thread["title"] == "Fix the header"


def test_a_second_import_adds_nothing_and_reports_up_to_date(claude_home):
    import_claude_chats()
    summary = import_claude_chats()
    assert summary.chats == 1
    assert summary.new_chats == 0
    assert summary.messages == 0
    assert len(studio_db.list_chat_projects()) == 1
    assert len(studio_db.list_chat_threads()) == 1


def test_turns_claude_appended_reach_an_already_imported_chat(claude_home):
    import_claude_chats()
    write_session(
        claude_home,
        "-Users-me-app",
        "session-one",
        [
            user_record("u1", "Fix the header", T0),
            assistant_record("a1", [text_block("Fixed it.")], T1, parent = "u1"),
            user_record("u2", "And the footer", T2, parent = "a1"),
        ],
    )
    summary = import_claude_chats()
    texts = [
        m["content"][0]["text"] for m in studio_db.list_chat_messages(thread_id_for("session-one"))
    ]
    assert texts == ["Fix the header", "Fixed it.", "And the footer"]
    assert summary.messages == 1
    assert summary.new_chats == 0


def test_a_message_edited_here_is_not_overwritten(claude_home):
    import_claude_chats()
    thread_id = thread_id_for("session-one")
    first = studio_db.list_chat_messages(thread_id)[0]
    studio_db.upsert_chat_message(
        {**first, "content": [{"type": "text", "text": "Fix the header, carefully"}]}
    )
    import_claude_chats()
    assert studio_db.list_chat_messages(thread_id)[0]["content"] == [
        {"type": "text", "text": "Fix the header, carefully"}
    ]


def test_a_message_deleted_here_is_not_recreated(claude_home):
    import_claude_chats()
    thread_id = thread_id_for("session-one")
    kept = studio_db.list_chat_messages(thread_id)[:1]
    studio_db.sync_chat_messages(thread_id, kept, prune_missing = True)
    summary = import_claude_chats()
    assert len(studio_db.list_chat_messages(thread_id)) == 1
    assert summary.messages == 0


def test_a_chat_moved_out_of_its_project_stays_where_it_was_put(claude_home):
    import_claude_chats()
    thread_id = thread_id_for("session-one")
    studio_db.update_chat_thread(thread_id, {"projectId": None})
    import_claude_chats()
    assert studio_db.get_chat_thread(thread_id)["projectId"] is None


def test_a_chat_deleted_here_is_not_resurrected(claude_home):
    # A second conversation stays, so this is a targeted delete rather than
    # an empty Studio, which Import from Claude Code treats as a blank slate.
    write_session(claude_home, "-Users-me-app", "session-two", [user_record("u9", "Keep me", T0)])
    import_claude_chats()
    thread_id = thread_id_for("session-one")
    studio_db.delete_chat_threads([thread_id])
    summary = import_claude_chats()
    assert summary.skipped == 1
    assert studio_db.get_chat_thread(thread_id) is None
    assert studio_db.get_chat_thread(thread_id_for("session-two")) is not None


def test_clearing_history_lets_claude_be_imported_again(claude_home):
    write_session(claude_home, "-Users-me-app", "session-two", [user_record("u9", "Keep me", T0)])
    import_claude_chats()
    studio_db.clear_chat_history()
    summary = import_claude_chats()
    assert summary.new_chats == 2
    assert studio_db.get_chat_thread(thread_id_for("session-one")) is not None
    assert studio_db.get_chat_thread(thread_id_for("session-two")) is not None


def test_a_late_tool_result_reaches_an_already_imported_call(claude_home):
    write_session(
        claude_home,
        "-Users-me-app",
        "session-one",
        [
            user_record("u1", "read it", T0),
            assistant_record(
                "a1",
                [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.txt"}}],
                T1,
                parent = "u1",
            ),
        ],
    )
    import_claude_chats()
    write_session(
        claude_home,
        "-Users-me-app",
        "session-one",
        [
            user_record("u1", "read it", T0),
            assistant_record(
                "a1",
                [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.txt"}}],
                T1,
                parent = "u1",
            ),
            tool_result_record("u2", "toolu_1", "file body", T2, parent = "a1"),
            assistant_record("a2", [text_block("Done.")], T3, parent = "u2"),
        ],
    )
    import_claude_chats()
    messages = studio_db.list_chat_messages(thread_id_for("session-one"))
    call = next(part for part in messages[1]["content"] if part.get("type") == "tool-call")
    assert call["result"] == "file body"
    assert messages[-1]["content"][0]["text"] == "Done."


def test_an_append_on_an_earlier_branch_still_arrives(claude_home):
    # A DFS flatten would insert the new turn in the middle of the list, so
    # the ledger slice would miss it and resync an old tail instead.
    branched = [
        user_record("u1", "try this", T0),
        assistant_record("a1", [text_block("first")], T1, parent = "u1"),
        user_record("u2", "no", T2, parent = "a1"),
        assistant_record("a2", [text_block("abandoned")], T3, parent = "u2"),
        user_record("u3", "retry", T4, parent = "a1"),
        assistant_record("a3", [text_block("kept")], T5, parent = "u3"),
    ]
    write_session(claude_home, "-Users-me-app", "session-one", branched)
    import_claude_chats()
    write_session(
        claude_home,
        "-Users-me-app",
        "session-one",
        [*branched, user_record("u4", "continue first branch", T6, parent = "a2")],
    )

    summary = import_claude_chats()

    messages = studio_db.list_chat_messages(thread_id_for("session-one"))
    assert summary.messages == 1
    assert messages[-1]["content"][0]["text"] == "continue first branch"
    abandoned = next(
        message for message in messages if message["content"][0].get("text") == "abandoned"
    )
    assert messages[-1]["parentId"] == abandoned["id"]


def test_a_no_op_import_does_not_promote_the_project_in_the_sidebar(claude_home):
    import_claude_chats()
    project_id = project_id_for("-Users-me-app")
    original = studio_db.get_chat_project(project_id)["updatedAt"]
    import_claude_chats()
    assert studio_db.get_chat_project(project_id)["updatedAt"] == original


def test_a_deleted_empty_project_is_not_recreated_for_moved_chats(claude_home):
    import_claude_chats()
    thread_id = thread_id_for("session-one")
    project_id = project_id_for("-Users-me-app")
    studio_db.update_chat_thread(thread_id, {"projectId": None})
    studio_db.delete_chat_project(project_id)
    import_claude_chats()
    assert studio_db.get_chat_project(project_id) is None
    assert studio_db.get_chat_thread(thread_id)["projectId"] is None


def test_a_chat_no_longer_in_studio_is_imported_whole_again(claude_home):
    # The ledger outlives the chats it describes -- a cleared history, a rolled
    # back database -- and must not leave those conversations half imported.
    studio_db.record_external_import_mark("claude", "session-one", 2**62, 99)
    summary = import_claude_chats()
    assert summary.messages == 2
    assert len(studio_db.list_chat_messages(thread_id_for("session-one"))) == 2


def test_an_empty_session_leaves_nothing_behind(claude_home, tmp_path):
    write_session(claude_home, "-Users-me-empty", "session-empty", [])
    summary = import_claude_chats()
    assert summary.projects == 1
    names = {p["name"] for p in studio_db.list_chat_projects()}
    assert "Claude · Users/me/empty" not in names


# Routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(claude_import_router, prefix = "/api/import")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_status_reports_what_is_waiting(client, claude_home):
    body = client.get("/api/import/claude/status").json()
    assert body == {"available": True, "projects": 1, "chats": 1}


def test_status_says_nothing_is_available_without_claude(client, tmp_path, monkeypatch):
    monkeypatch.setenv(discovery.CLAUDE_HOME_ENV, str(tmp_path / "no-claude-here"))
    body = client.get("/api/import/claude/status").json()
    assert body == {"available": False, "projects": 0, "chats": 0}


def test_the_import_endpoint_reports_what_it_wrote(client, claude_home):
    body = client.post("/api/import/claude").json()
    assert body == {
        "projects": 1,
        "chats": 1,
        "new_chats": 1,
        "messages": 2,
        "skipped": 0,
        "warnings": [],
    }


def test_the_import_endpoint_needs_a_signed_in_user(claude_home):
    app = FastAPI()
    app.include_router(claude_import_router, prefix = "/api/import")
    assert TestClient(app).post("/api/import/claude").status_code in (401, 403)
