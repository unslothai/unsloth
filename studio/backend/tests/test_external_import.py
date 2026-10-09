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
