# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-chat file history: what python, terminal and edit_file change is recorded per turn, and
/rewind puts back exactly this chat's changes."""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import file_history, tools
from core.inference.tools import execute_tool


@pytest.fixture
def chat(tmp_path, monkeypatch):
    workdir = tmp_path / "sandbox"
    workdir.mkdir()
    history = file_history.FileHistory(str(tmp_path / "file_history"))
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    monkeypatch.setattr(file_history, "store", lambda: history)
    yield workdir, history
    file_history.set_turn(None, None)


def _walk(workdir):
    return tools._history_snapshot(tools._snapshot_workdir_files(workdir))


def _rewind(history, turn_ids, **kwargs):
    return history.rewind("chat-1", turn_ids, _walk, **kwargs)


def _edit(path, old, new):
    return execute_tool(
        "edit_file",
        {"path": path, "edits": [{"old_string": old, "new_string": new}]},
        session_id = "s",
    )


def _terminal(command):
    return execute_tool("terminal", {"command": command}, session_id = "s")


def test_edit_file_turns_rewind_in_order(chat):
    workdir, history = chat
    (workdir / "keep.txt").write_text("original\n")
    file_history.set_turn("chat-1", "u1")
    _edit("app.py", "", "x = 1\n")
    _edit("keep.txt", "original", "changed")
    file_history.set_turn("chat-1", "u2")
    _edit("app.py", "x = 1", "x = 2")

    report = _rewind(history, ["u2"])
    assert report.restored == ["app.py"]
    assert (workdir / "app.py").read_text() == "x = 1\n"
    assert (workdir / "keep.txt").read_text() == "changed\n"

    report = _rewind(history, ["u1", "u2"])
    assert report.deleted == ["app.py"] and report.restored == ["keep.txt"]
    assert not (workdir / "app.py").exists()
    assert (workdir / "keep.txt").read_text() == "original\n"


@pytest.mark.skipif(os.name != "posix", reason = "uses a POSIX shell")
def test_terminal_and_python_writes_and_deletes_are_restored(chat):
    workdir, history = chat
    (workdir / "data.csv").write_text("a,b\n")
    (workdir / "old.txt").write_text("bye\n")
    file_history.set_turn("chat-1", "u1")
    # `rm` waits for approval in the terminal tool; python deletes without asking.
    _terminal("printf 'x' >> data.csv && printf hi > new.txt")
    execute_tool("python", {"code": "import os; os.remove('old.txt')"}, session_id = "s")
    assert (workdir / "data.csv").read_text() == "a,b\nx"

    report = _rewind(history, ["u1"])
    assert report.restored == ["data.csv"]
    assert report.deleted == ["new.txt"]
    assert report.recreated == ["old.txt"]
    assert (workdir / "data.csv").read_text() == "a,b\n"
    assert (workdir / "old.txt").read_text() == "bye\n"
    assert not (workdir / "new.txt").exists()


def test_a_dry_run_changes_nothing_and_a_second_rewind_is_a_no_op(chat):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("a.txt", "", "a\n")
    preview = _rewind(history, ["u1"], dry_run = True)
    assert preview.has_changes and preview.deleted == ["a.txt"] and preview.undo_turn_id is None
    assert (workdir / "a.txt").exists()
    assert _rewind(history, ["u1"]).deleted == ["a.txt"]
    again = _rewind(history, ["u1"])
    assert again.has_changes and not (again.deleted or again.restored or again.recreated)


def test_a_file_changed_since_is_skipped_unless_forced(chat):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("notes.md", "", "mine\n")
    (workdir / "notes.md").write_text("the user's own edit\n")
    report = _rewind(history, ["u1"])
    assert report.conflicts == ["notes.md"]
    assert (workdir / "notes.md").read_text() == "the user's own edit\n"
    assert _rewind(history, ["u1"], force = True).deleted == ["notes.md"]


def test_a_rewind_can_be_undone(chat):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("a.txt", "", "one\n")
    file_history.set_turn("chat-1", "u2")
    _edit("a.txt", "one", "two")
    report = _rewind(history, ["u2"])
    assert (workdir / "a.txt").read_text() == "one\n"
    undo = _rewind(history, [report.undo_turn_id])
    assert undo.restored == ["a.txt"]
    assert (workdir / "a.txt").read_text() == "two\n"


def test_nothing_is_recorded_outside_a_chat_turn(chat):
    workdir, history = chat
    _edit("a.txt", "", "a\n")
    assert history._db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
    assert not _rewind(history, ["u1"]).has_changes


def test_another_chats_changes_are_never_touched(chat):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("mine.txt", "", "1\n")
    file_history.set_turn("chat-2", "v1")
    _edit("theirs.txt", "", "2\n")
    assert _rewind(history, ["u1"]).deleted == ["mine.txt"]
    assert (workdir / "theirs.txt").read_text() == "2\n"


def test_a_rewind_waits_for_a_running_call(chat):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("a.txt", "", "a\n")
    token = history.begin("chat-1", str(workdir), "u2", _walk(str(workdir))[0])
    with pytest.raises(file_history.RewindBusy):
        _rewind(history, ["u1"])
    history.end(token, _walk(str(workdir))[0])
    assert _rewind(history, ["u1"]).deleted == ["a.txt"]


def test_deleting_the_chat_drops_its_history(chat):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("a.txt", "", "first version\n")
    _edit("a.txt", "first version", "second version")
    stored = history._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    history.delete_chat("chat-1")
    assert history._db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
    # Only the folder's current image is still referenced.
    assert history._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] < stored


def test_large_files_are_reported_not_restored(chat, monkeypatch):
    workdir, history = chat
    monkeypatch.setattr(file_history, "MAX_FILE_BYTES", 8)
    (workdir / "big.bin").write_bytes(b"0123456789")
    file_history.set_turn("chat-1", "u1")
    _edit("note.txt", "", "n\n")  # takes the baseline, with big.bin over the cap
    file_history.set_turn("chat-1", "u2")
    if os.name == "posix":
        _terminal("printf 'abcdefghijkl' > big.bin")
        report = _rewind(history, ["u2"])
        assert report.untracked == ["big.bin"]
        assert (workdir / "big.bin").read_bytes() == b"abcdefghijkl"


def test_the_route_previews_rewinds_and_refuses_while_generating(chat, monkeypatch):
    from fastapi import HTTPException

    from routes import chat_history
    from state import active_generations

    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("a.txt", "", "a\n")
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda thread_id: {"id": thread_id})
    monkeypatch.setattr(active_generations, "active_thread_ids", lambda scope = None: set())

    def call(**body):
        request = chat_history.ChatRewindRequest(turn_ids = ["u1"], **body)
        return chat_history.rewind_thread_files("chat-1", request, current_subject = "user")

    preview = call(dry_run = True)
    assert (
        preview["has_changes"]
        and preview["deleted"] == ["a.txt"]
        and preview["undo_turn_id"] is None
    )
    assert (workdir / "a.txt").exists()

    monkeypatch.setattr(active_generations, "active_thread_ids", lambda scope = None: {"chat-1"})
    with pytest.raises(HTTPException) as refused:
        call()
    assert refused.value.status_code == 409

    monkeypatch.setattr(active_generations, "active_thread_ids", lambda scope = None: set())
    done = call()
    assert done["deleted"] == ["a.txt"] and done["undo_turn_id"]
    assert not (workdir / "a.txt").exists()
