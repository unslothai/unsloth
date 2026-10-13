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
    (workdir / "keep.txt").write_text("original\n", encoding = "utf-8")
    file_history.set_turn("chat-1", "u1")
    _edit("app.py", "", "x = 1\n")
    _edit("keep.txt", "original", "changed")
    file_history.set_turn("chat-1", "u2")
    _edit("app.py", "x = 1", "x = 2")

    report = _rewind(history, ["u2"])
    assert report.restored == ["app.py"]
    assert (workdir / "app.py").read_text(encoding = "utf-8") == "x = 1\n"
    assert (workdir / "keep.txt").read_text(encoding = "utf-8") == "changed\n"

    report = _rewind(history, ["u1", "u2"])
    assert report.deleted == ["app.py"] and report.restored == ["keep.txt"]
    assert not (workdir / "app.py").exists()
    assert (workdir / "keep.txt").read_text(encoding = "utf-8") == "original\n"


@pytest.mark.skipif(os.name != "posix", reason = "uses a POSIX shell")
def test_terminal_and_python_writes_and_deletes_are_restored(chat):
    workdir, history = chat
    (workdir / "data.csv").write_text("a,b\n", encoding = "utf-8")
    (workdir / "old.txt").write_text("bye\n", encoding = "utf-8")
    file_history.set_turn("chat-1", "u1")
    # `rm` waits for approval in the terminal tool; python deletes without asking.
    _terminal("printf 'x' >> data.csv && printf hi > new.txt")
    execute_tool("python", {"code": "import os; os.remove('old.txt')"}, session_id = "s")
    assert (workdir / "data.csv").read_text(encoding = "utf-8") == "a,b\nx"

    report = _rewind(history, ["u1"])
    assert report.restored == ["data.csv"]
    assert report.deleted == ["new.txt"]
    assert report.recreated == ["old.txt"]
    assert (workdir / "data.csv").read_text(encoding = "utf-8") == "a,b\n"
    assert (workdir / "old.txt").read_text(encoding = "utf-8") == "bye\n"
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
    (workdir / "notes.md").write_text("the user's own edit\n", encoding = "utf-8")
    report = _rewind(history, ["u1"])
    assert report.conflicts == ["notes.md"]
    assert (workdir / "notes.md").read_text(encoding = "utf-8") == "the user's own edit\n"
    assert _rewind(history, ["u1"], force = True).deleted == ["notes.md"]


def test_a_rewind_can_be_undone(chat):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("a.txt", "", "one\n")
    file_history.set_turn("chat-1", "u2")
    _edit("a.txt", "one", "two")
    report = _rewind(history, ["u2"])
    assert (workdir / "a.txt").read_text(encoding = "utf-8") == "one\n"
    undo = _rewind(history, [report.undo_turn_id])
    assert undo.restored == ["a.txt"]
    assert (workdir / "a.txt").read_text(encoding = "utf-8") == "two\n"


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
    assert (workdir / "theirs.txt").read_text(encoding = "utf-8") == "2\n"


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


def test_a_walk_cut_short_by_the_folder_cap_is_not_complete(chat, monkeypatch):
    workdir, _history = chat
    for name in ("a", "b", "c"):
        (workdir / name).mkdir()
        (workdir / name / "f.txt").write_text(name, encoding = "utf-8")
    assert _walk(str(workdir))[1] is True
    # Fewer files than the file cap, so only the folder cap can say the walk missed some.
    monkeypatch.setattr(tools, "_MAX_SNAPSHOT_DIRS", 2)
    assert _walk(str(workdir))[1] is False


def test_a_damaged_copy_is_one_files_error_not_a_failed_rewind(chat):
    workdir, history = chat
    (workdir / "a.txt").write_text("a" * 4096, encoding = "utf-8")
    (workdir / "b.txt").write_text("b" * 4096, encoding = "utf-8")
    file_history.set_turn("chat-1", "u1")
    _edit("a.txt", "a" * 4096, "x")
    _edit("b.txt", "b" * 4096, "y")
    for blob in Path(history.objects).glob("??/*"):
        payload = blob.read_bytes()
        if payload[:1] != b"R":
            blob.write_bytes(payload[:1] + b"\0garbage")
    report = _rewind(history, ["u1"])
    assert report.errors and all("damaged" in error for error in report.errors)


def test_the_cap_counts_history_not_the_folders_current_files(chat, monkeypatch):
    workdir, history = chat
    (workdir / "big.txt").write_text("z" * 1000, encoding = "utf-8")
    file_history.set_turn("chat-1", "u1")
    _edit("app.py", "", "x = 1\n")
    monkeypatch.setattr(file_history, "GLOBAL_CAP_BYTES", 1)
    history.gc()
    # Only the current images (big.txt, app.py) are over the cap, and they cannot be dropped.
    assert history._db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] >= 1


def test_edit_file_still_works_when_history_cannot_open(chat, monkeypatch):
    workdir, _history = chat

    def broken():
        raise OSError("read-only")

    monkeypatch.setattr(file_history, "store", broken)
    file_history.set_turn("chat-1", "u1")
    _edit("app.py", "", "x = 1\n")
    assert (workdir / "app.py").read_text(encoding = "utf-8") == "x = 1\n"


def test_a_file_written_while_another_chats_call_ran_is_a_conflict(chat):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("seed.txt", "", "s\n")  # takes the folder's baseline
    mine = history.begin("chat-1", str(workdir), "u2", _walk(str(workdir))[0])
    theirs = history.begin("chat-2", str(workdir), "v1", _walk(str(workdir))[0])
    (workdir / "theirs.txt").write_text("2\n", encoding = "utf-8")
    history.end(mine, _walk(str(workdir))[0])
    history.end(theirs, _walk(str(workdir))[0])
    report = _rewind(history, ["u2"])
    assert report.conflicts == ["theirs.txt"] and report.deleted == []
    assert (workdir / "theirs.txt").read_text(encoding = "utf-8") == "2\n"


@pytest.mark.skipif(os.name != "posix", reason = "uses a symlink")
def test_a_folder_swapped_for_a_link_is_never_written_through(chat, tmp_path):
    workdir, history = chat
    file_history.set_turn("chat-1", "u1")
    _edit("a.txt", "", "1\n")
    _edit("a.txt", "1", "2")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "a.txt").write_text("2\n", encoding = "utf-8")
    workdir.rename(tmp_path / "moved")
    workdir.symlink_to(elsewhere, target_is_directory = True)
    report = history.rewind(
        "chat-1", ["u1"], lambda wd: tools._history_snapshot(tools._snapshot_workdir_files(wd))
    )
    assert report.errors and (elsewhere / "a.txt").read_text(encoding = "utf-8") == "2\n"


def test_deleting_a_chat_that_only_read_files_releases_its_baseline(chat):
    workdir, history = chat
    (workdir / "data.csv").write_text("a,b\n", encoding = "utf-8")
    file_history.set_turn("chat-1", "u1")
    execute_tool("python", {"code": "print(open('data.csv').read())"}, session_id = "s")
    assert history._db.execute("SELECT COUNT(*) FROM workdir_state").fetchone()[0] == 1
    history.delete_chat("chat-1")
    assert history._db.execute("SELECT COUNT(*) FROM workdir_state").fetchone()[0] == 0
    assert history._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 0


def test_an_edit_missed_by_a_capped_first_walk_restores_the_original(chat, monkeypatch):
    workdir, history = chat
    (workdir / "a").mkdir()
    (workdir / "z").mkdir()
    (workdir / "z" / "keep.txt").write_text("original\n", encoding = "utf-8")
    monkeypatch.setattr(tools, "_MAX_SNAPSHOT_DIRS", 1)
    file_history.set_turn("chat-1", "u1")
    _edit("z/keep.txt", "original", "changed")
    report = _rewind(history, ["u1"])
    assert report.restored == ["z/keep.txt"] and report.deleted == []
    assert (workdir / "z" / "keep.txt").read_text(encoding = "utf-8") == "original\n"
