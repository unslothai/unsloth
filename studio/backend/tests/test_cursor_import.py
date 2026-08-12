# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Importing Cursor's agent conversations into Studio."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import storage.studio_db as studio_db
from auth.authentication import get_current_subject
from core.cursor_import import discovery, import_cursor_chats, project_id_for, thread_id_for
from core.cursor_import.transcripts import read_transcript
from routes.cursor_import import router as cursor_import_router


def write_transcript(home: Path, slug: str, session_id: str, records: list[dict]) -> Path:
    """Lay out one session the way Cursor does, and return its file."""
    session_dir = home / "projects" / slug / "agent-transcripts" / session_id
    session_dir.mkdir(parents = True, exist_ok = True)
    path = session_dir / f"{session_id}.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding = "utf-8",
    )
    return path


def turn(role: str, text: str) -> dict:
    return {"role": role, "message": {"content": [{"type": "text", "text": text}]}}


@pytest.fixture
def cursor_home(tmp_path, monkeypatch):
    """A Cursor state directory with one project and one conversation in it."""
    home = tmp_path / "cursor"
    write_transcript(
        home,
        "Users-me-app",
        "session-one",
        [
            turn("user", "<user_query>Fix the header</user_query>"),
            turn("assistant", "Fixed it."),
        ],
    )
    monkeypatch.setenv(discovery.CURSOR_HOME_ENV, str(home))
    return home


# Discovery


def test_lists_a_project_with_its_conversations(cursor_home):
    workspaces = discovery.list_cursor_workspaces()

    assert [(workspace.slug, len(workspace.transcripts)) for workspace in workspaces] == [
        ("Users-me-app", 1)
    ]


def test_skips_a_state_directory_holding_no_conversation(cursor_home):
    (cursor_home / "projects" / "Users-me-empty").mkdir(parents = True)

    assert [w.slug for w in discovery.list_cursor_workspaces()] == ["Users-me-app"]


def test_skips_cursor_own_bookkeeping_directories(cursor_home):
    write_transcript(cursor_home, ".internal", "session-two", [turn("user", "hi")])

    assert [w.slug for w in discovery.list_cursor_workspaces()] == ["Users-me-app"]


def test_counts_subagent_transcripts_without_importing_them(cursor_home):
    session_dir = cursor_home / "projects" / "Users-me-app" / "agent-transcripts" / "session-one"
    subagents = session_dir / "subagents"
    subagents.mkdir()
    (subagents / "delegated.jsonl").write_text("{}\n", encoding = "utf-8")

    workspace = discovery.list_cursor_workspaces()[0]

    assert workspace.subagent_transcripts == 1
    assert len(workspace.transcripts) == 1


def test_names_a_project_after_the_folder_it_resolves_to(tmp_path, monkeypatch):
    home = tmp_path / "cursor"
    project = tmp_path / "work" / "checkout"
    project.mkdir(parents = True)
    write_transcript(home, discovery.state_slug(project), "s1", [turn("user", "hi")])
    monkeypatch.setenv(discovery.CURSOR_HOME_ENV, str(home))

    assert discovery.list_cursor_workspaces()[0].name == "checkout"


def test_names_a_project_from_its_slug_when_the_folder_is_gone(cursor_home, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/Users/me")))
    write_transcript(cursor_home, "Users-me-deleted-long-ago", "s2", [turn("user", "hi")])

    names = {w.slug: w.name for w in discovery.list_cursor_workspaces()}

    # The folder is gone, so the home prefix comes off and what identifies the
    # project stays. Taking the last token alone would call this one "ago".
    assert names["Users-me-deleted-long-ago"] == "deleted-long-ago"


def test_names_the_no_folder_window_for_what_it_is(cursor_home):
    write_transcript(cursor_home, discovery.NO_FOLDER_SLUG, "s3", [turn("user", "hi")])

    names = {w.slug: w.name for w in discovery.list_cursor_workspaces()}

    assert names[discovery.NO_FOLDER_SLUG] == "No folder open"


def test_resolves_a_slug_to_the_one_folder_that_exists(tmp_path):
    project = tmp_path / "a-b" / "c"
    project.mkdir(parents = True)

    assert discovery.resolve_state_slug(discovery.state_slug(project)) == project


def test_refuses_to_guess_between_two_readings_of_a_slug(tmp_path):
    (tmp_path / "a-b" / "c").mkdir(parents = True)
    (tmp_path / "a" / "b-c").mkdir(parents = True)

    slug = discovery.state_slug(tmp_path / "a-b" / "c")

    assert discovery.resolve_state_slug(slug) is None


# Transcript parsing


def test_keeps_the_prompt_and_drops_the_context_cursor_injected(tmp_path):
    path = write_transcript(
        tmp_path,
        "slug",
        "s1",
        [turn("user", "<attached_files>noise</attached_files>\n<user_query>Ship it</user_query>")],
    )

    transcript = read_transcript(path, "thread-1")

    assert transcript.messages[0]["content"] == [{"type": "text", "text": "Ship it"}]


def test_imports_a_tool_call_the_ui_can_render(tmp_path):
    path = write_transcript(
        tmp_path,
        "slug",
        "s1",
        [
            turn("user", "read it"),
            {
                "role": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {"path": "a.txt"}},
                    ]
                },
            },
        ],
    )

    transcript = read_transcript(path, "thread-1")
    call = transcript.messages[1]["content"][0]

    assert transcript.tool_calls == 1
    assert call["type"] == "tool-call"
    assert call["toolName"] == "Read"
    assert call["args"] == {"path": "a.txt"}


def test_passes_over_session_events_and_a_half_written_line(tmp_path):
    path = write_transcript(tmp_path, "slug", "s1", [turn("user", "hi"), {"type": "turn_ended"}])
    with path.open("a", encoding = "utf-8") as handle:
        handle.write('{"role": "assis')

    transcript = read_transcript(path, "thread-1")

    assert len(transcript.messages) == 1
    assert transcript.skipped_records == 2


def test_titles_a_conversation_from_its_first_prompt(tmp_path):
    path = write_transcript(
        tmp_path, "slug", "s1", [turn("user", "Fix the header\nand the footer")]
    )

    assert read_transcript(path, "thread-1").title == "Fix the header"


# Importing


def _threads_of(project_id: str) -> list[dict]:
    return [t for t in studio_db.list_chat_threads() if t["projectId"] == project_id]


def test_imports_a_conversation_into_a_project_named_after_cursor(cursor_home):
    summary = import_cursor_chats()

    project = studio_db.get_chat_project(project_id_for("Users-me-app"))
    threads = _threads_of(project["id"])
    assert summary.projects == 1
    assert summary.chats == 1
    assert summary.new_chats == 1
    assert summary.messages == 2
    assert project["name"] == "Cursor · Users-me-app"
    assert [thread["title"] for thread in threads] == ["Fix the header"]
    assert len(studio_db.list_chat_messages(threads[0]["id"])) == 2


def test_a_second_import_updates_rather_than_duplicates(cursor_home):
    import_cursor_chats()
    summary = import_cursor_chats()

    assert summary.chats == 1
    # Nothing new arrived, which is what lets the UI say "up to date".
    assert summary.new_chats == 0
    assert len(studio_db.list_chat_projects()) == 1
    assert len(studio_db.list_chat_threads()) == 1


def test_a_second_import_brings_in_only_what_cursor_added(cursor_home):
    import_cursor_chats()
    write_transcript(cursor_home, "Users-me-app", "session-two", [turn("user", "Another one")])

    summary = import_cursor_chats()

    assert summary.chats == 2
    assert summary.new_chats == 1


def test_a_renamed_chat_keeps_the_name_the_user_gave_it(cursor_home):
    import_cursor_chats()
    thread_id = thread_id_for("session-one")
    studio_db.update_chat_thread(thread_id, {"title": "Header work"})

    import_cursor_chats()

    assert studio_db.get_chat_thread(thread_id)["title"] == "Header work"


def test_a_renamed_project_keeps_the_name_the_user_gave_it(cursor_home):
    import_cursor_chats()
    project_id = project_id_for("Users-me-app")
    studio_db.update_chat_project(project_id, {"name": "The app"})

    import_cursor_chats()

    assert studio_db.get_chat_project(project_id)["name"] == "The app"


def test_a_chat_deleted_after_an_import_is_not_brought_back(cursor_home):
    import_cursor_chats()
    thread_id = thread_id_for("session-one")
    studio_db.delete_chat_threads([thread_id])

    summary = import_cursor_chats()

    assert studio_db.get_chat_thread(thread_id) is None
    assert summary.chats == 0
    assert summary.skipped == 1


def test_a_session_shared_with_the_no_folder_window_lands_in_the_real_project(cursor_home):
    # Cursor files a session that predates opening a folder under both names.
    shared = [turn("user", "Started before opening the folder")]
    write_transcript(cursor_home, "Users-me-app", "shared-session", shared)
    write_transcript(cursor_home, discovery.NO_FOLDER_SLUG, "shared-session", shared)

    summary = import_cursor_chats()

    thread = studio_db.get_chat_thread(thread_id_for("shared-session"))
    assert thread["projectId"] == project_id_for("Users-me-app")
    # One conversation, imported once, and no empty project left behind for the
    # window that only held a copy of it.
    assert summary.chats == 2
    assert len(studio_db.list_chat_projects()) == 1


def test_an_empty_conversation_is_left_out(cursor_home):
    write_transcript(cursor_home, "Users-me-app", "empty-session", [{"type": "turn_ended"}])

    summary = import_cursor_chats()

    assert summary.chats == 1
    assert summary.skipped == 1


def test_a_project_whose_only_conversation_is_empty_leaves_nothing_behind(tmp_path, monkeypatch):
    home = tmp_path / "cursor"
    write_transcript(home, "Users-me-blank", "blank-session", [{"type": "turn_ended"}])
    monkeypatch.setenv(discovery.CURSOR_HOME_ENV, str(home))

    summary = import_cursor_chats()

    assert summary.chats == 0
    # No empty project in the sidebar for a Cursor folder that held nothing.
    assert studio_db.list_chat_projects() == []


def test_nothing_to_import_is_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv(discovery.CURSOR_HOME_ENV, str(tmp_path / "no-cursor-here"))

    summary = import_cursor_chats()

    assert summary.chats == 0
    assert summary.imported_anything is False


# Routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(cursor_import_router, prefix = "/api/import")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_status_reports_what_is_waiting(client, cursor_home):
    body = client.get("/api/import/cursor/status").json()

    assert body == {"available": True, "projects": 1, "chats": 1}


def test_status_says_nothing_is_available_without_cursor(client, tmp_path, monkeypatch):
    monkeypatch.setenv(discovery.CURSOR_HOME_ENV, str(tmp_path / "no-cursor-here"))

    body = client.get("/api/import/cursor/status").json()

    assert body == {"available": False, "projects": 0, "chats": 0}


def test_the_import_endpoint_reports_what_it_wrote(client, cursor_home):
    body = client.post("/api/import/cursor").json()

    assert body == {
        "projects": 1,
        "chats": 1,
        "new_chats": 1,
        "messages": 2,
        "skipped": 0,
        "warnings": [],
    }


def test_the_import_endpoint_needs_a_signed_in_user(cursor_home):
    app = FastAPI()
    app.include_router(cursor_import_router, prefix = "/api/import")

    assert TestClient(app).post("/api/import/cursor").status_code in (401, 403)
