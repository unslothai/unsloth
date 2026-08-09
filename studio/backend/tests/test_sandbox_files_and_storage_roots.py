# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a chat's files live, and how a user gets them back.

Every assertion here failed before: the sandbox ignored UNSLOTH_STUDIO_HOME, only
images could be fetched, nothing listed a chat's files, bash reported none, the
compiled cache landed in the launcher's CWD, and a deleted chat left its folder
behind. Verified on Windows, macOS and Linux.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import platform
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Sandbox location
# ---------------------------------------------------------------------------
def test_sandbox_lives_under_the_studio_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "userprofile"
    studio_home = tmp_path / "custom_studio_home"
    fake_home.mkdir()
    studio_home.mkdir()

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))

    from core.inference import tools

    tools._workdirs.clear()
    wd = Path(tools.get_sandbox_workdir("__LOCALID_aB3xY7q"))
    print(f"\n[{platform.system()}] sandbox workdir = {wd}")

    # It follows UNSLOTH_STUDIO_HOME instead of dropping a third folder in the
    # user's home next to .unsloth.
    assert str(wd).startswith(str(studio_home)), wd
    assert not str(wd).startswith(str(fake_home / "studio_sandbox")), wd
    assert wd.parent.name == "sandbox"
    assert wd.name.startswith("__LOCALID_")


def test_sandbox_home_override(tmp_path, monkeypatch):
    """A user who wants the sandbox on another volume can say so."""
    elsewhere = tmp_path / "other volume" / "sandboxes"
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(elsewhere))

    from core.inference import tools

    tools._workdirs.clear()
    wd = Path(tools.get_sandbox_workdir("__LOCALID_aB3xY7q"))
    assert wd == elsewhere / "__LOCALID_aB3xY7q"


def test_legacy_sandbox_is_migrated(tmp_path, monkeypatch):
    """Files created before the move are carried over, not abandoned."""
    fake_home = tmp_path / "userprofile"
    studio_home = tmp_path / "studio_home"
    legacy = fake_home / "studio_sandbox" / "__LOCALID_old1234"
    legacy.mkdir(parents = True)
    (legacy / "results.csv").write_text("a,b\n1,2\n")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    wd = Path(tools.get_sandbox_workdir("__LOCALID_new5678"))
    # Another chat's folder rides the background pass, so a first call never
    # waits on the whole tree.
    for thread in threading.enumerate():
        if thread.name == "sandbox-migrate":
            thread.join(30)
    moved = wd.parent / "__LOCALID_old1234" / "results.csv"
    print(f"\nmigrated to {moved}")
    assert moved.is_file()
    assert moved.read_text() == "a,b\n1,2\n"


# ---------------------------------------------------------------------------
# 2. Getting files back out
# ---------------------------------------------------------------------------
def test_images_stay_inline_and_everything_else_downloads():
    """Images keep a real media type; the rest are opaque attachments.

    The allowlist is deliberately NOT just widened: the model picks these
    filenames, so an inline text/html would be same-origin script execution.
    """
    from routes.inference import _SANDBOX_MEDIA_TYPES

    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        assert ext in _SANDBOX_MEDIA_TYPES
    for ext in (".csv", ".txt", ".py", ".json", ".pdf", ".zip", ".html", ".svg"):
        assert ext not in _SANDBOX_MEDIA_TYPES, ext

    import inspect
    from routes import inference

    src = inspect.getsource(inference.serve_sandbox_file)
    assert "application/octet-stream" in src
    assert "Content-Disposition" in src
    assert "attachment" in src
    assert "File type not allowed" not in src, "non-images are no longer rejected"


def test_sandbox_listing_route_exists():
    """The UI can enumerate a chat's files, and learn where they live."""
    from routes.inference import router

    sandbox_routes = sorted(r.path for r in router.routes if "sandbox" in r.path)
    print(f"\nsandbox routes = {sandbox_routes}")
    # :path so a file written into a subdirectory is reachable.
    assert sandbox_routes == ["/sandbox/{session_id}", "/sandbox/{session_id}/{filename:path}"]

    import inspect
    from routes import inference

    src = inspect.getsource(inference.list_sandbox_files)
    assert '"path"' in src, "the listing must answer 'where did my file go'"


# ---------------------------------------------------------------------------
# 3. Reporting what a call created
# ---------------------------------------------------------------------------
def test_both_executors_report_created_files(tmp_path, monkeypatch):
    """A file is reported whether it came from python or from bash."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_exec123"
    workdir = tools.get_sandbox_workdir(session)

    for name, run in (
        (
            "py.csv",
            lambda: tools._python_exec("open('py.csv','w').write('a,b\\n')", session_id = session),
        ),
        ("sh.csv", lambda: tools._bash_exec("printf 'a,b\\n' > sh.csv", session_id = session)),
    ):
        result = run()
        print(f"\n{name} -> {result!r}")
        assert "__FILES__:" in result, result
        assert name in result
        assert os.path.isfile(os.path.join(workdir, name))

    # The sentinel never reaches the model.
    from core.inference.tool_loop_controller import strip_result_for_model

    stripped = strip_result_for_model('done\n__FILES__:[{"name": "x.csv"}]')
    assert "__FILES__" not in stripped
    assert stripped.strip() == "done"


def test_internal_temp_files_are_not_reported(tmp_path, monkeypatch):
    """The executor's own scratch script is not a user-facing artifact."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    result = tools._python_exec("print('hi')", session_id = "__LOCALID_tmp999")
    assert "studio_exec_" not in result
    assert "__FILES__" not in result


def test_tool_description_says_files_are_kept():
    from core.inference import tools

    note = tools._SANDBOX_PATHS_NOTE
    print(f"\nsandbox note = {note!r}")
    assert "download link" in note
    assert "name the files you created" in note
    assert "absolute path" in note


# ---------------------------------------------------------------------------
# 4. Compiled cache
# ---------------------------------------------------------------------------
def test_compiled_cache_is_pinned_under_the_studio_home(tmp_path, monkeypatch):
    """Not left CWD-relative, which put it in %USERPROFILE% on Windows."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    import importlib
    from utils.paths import storage_roots

    importlib.reload(storage_roots)
    storage_roots.ensure_studio_directories()

    location = Path(os.environ["UNSLOTH_COMPILE_LOCATION"])
    print(f"\n[{platform.system()}] compiled cache pinned to {location}")
    assert location.is_absolute()
    assert str(tmp_path / "studio_home") in str(location)


def test_cache_cleanup_finds_the_configured_and_cwd_caches(tmp_path, monkeypatch):
    """Cleanup can now see a cache created outside the source tree.

    Both cases matter: the pinned location for new installs, and the launcher's
    CWD for a machine that already has one sitting in the user profile.
    """
    from utils import cache_cleanup

    monkeypatch.chdir(tmp_path)
    configured = tmp_path / "studio_home" / "compiled_cache"
    configured.mkdir(parents = True)
    (tmp_path / "unsloth_compiled_cache").mkdir()
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(configured))

    found = {str(Path(d).resolve()) for d in cache_cleanup.get_existing_cache_dirs()}
    print("\nfound cache dirs:")
    for d in sorted(found):
        print("   ", d)
    assert str(configured.resolve()) in found
    assert str((tmp_path / "unsloth_compiled_cache").resolve()) in found


# ---------------------------------------------------------------------------
# 5. Cleanup on chat delete
# ---------------------------------------------------------------------------
def test_deleting_a_chat_cleans_up_its_sandbox(tmp_path, monkeypatch):
    """An empty sandbox always goes; files need an explicit opt-in."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    empty = Path(tools.get_sandbox_workdir("__LOCALID_empty11"))
    assert tools.remove_session_sandbox("__LOCALID_empty11") is True
    assert not empty.exists()

    tools._workdirs.clear()
    withfile = Path(tools.get_sandbox_workdir("__LOCALID_files22"))
    (withfile / "keep.csv").write_text("a\n")
    # Not deleted implicitly: those files are the user's.
    assert tools.remove_session_sandbox("__LOCALID_files22") is False
    assert (withfile / "keep.csv").is_file()
    assert tools.remove_session_sandbox("__LOCALID_files22", delete_files = True) is True
    assert not withfile.exists()


def test_sandbox_removal_cannot_escape_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    outside = tmp_path / "precious"
    outside.mkdir()
    for bad in ("..", "../precious", "/etc", "project-abc", ""):
        assert tools.remove_session_sandbox(bad, delete_files = True) is False
    assert outside.is_dir()


def test_a_windows_device_name_never_becomes_a_directory(tmp_path, monkeypatch):
    """CON, NUL and friends are reserved on Windows even as folder names, and
    the session id comes from the caller."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    seen = set()
    for reserved in ("con", "NUL", "aux", "COM1", "lpt9", "nul.txt"):
        workdir = Path(tools.get_sandbox_workdir(reserved))
        assert workdir.name.startswith("_id-"), reserved
        seen.add(workdir.name)
        assert tools.remove_session_sandbox(reserved, delete_files = True) is True
    assert len(seen) == 6, "reserved names shared a directory"


def test_reading_a_sandbox_never_creates_it(tmp_path, monkeypatch):
    """A GET must not leave a folder behind for every id it is asked about."""
    root = tmp_path / "sb"
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    resolved = Path(tools.resolve_sandbox_workdir("__LOCALID_ghost"))
    assert resolved == root / "__LOCALID_ghost"
    assert not resolved.exists()
    assert not root.exists()
    # The creating resolver still agrees on the path.
    assert Path(tools.get_sandbox_workdir("__LOCALID_ghost")) == resolved
    assert resolved.is_dir()


def test_clearing_the_compiled_cache_covers_the_configured_location(tmp_path, monkeypatch):
    """The cleanup must follow UNSLOTH_COMPILE_LOCATION, not just the defaults."""
    pinned = tmp_path / "home" / "compiled_cache"
    pinned.mkdir(parents = True)
    (pinned / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n")
    (pinned / "UnslothSFTTrainer.py").write_text("x = 1\n")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(pinned))

    from utils import cache_cleanup

    assert pinned in cache_cleanup.get_existing_cache_dirs()
    cache_cleanup.clear_unsloth_compiled_cache(preserve_patterns = ["Unsloth*Trainer.py"])
    assert not (pinned / "unsloth_compiled_module_gemma3.py").exists()
    assert (pinned / "UnslothSFTTrainer.py").is_file()


def test_a_file_in_a_subdirectory_is_reported_and_servable(tmp_path, monkeypatch):
    """`df.to_csv("outputs/report.csv")` is ordinary; a top-level listing saw
    only the directory and dropped it, so no chip and no download."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_nested1"))
    before = tools._snapshot_workdir_files(str(workdir))
    (workdir / "outputs").mkdir()
    (workdir / "outputs" / "report.csv").write_text("a,b\n")
    (workdir / "top.txt").write_text("x")

    sentinels = tools._created_file_sentinels(str(workdir), before)
    assert "outputs/report.csv" in sentinels
    assert "top.txt" in sentinels


def test_the_walk_is_bounded(tmp_path, monkeypatch):
    """A chat that unpacked an archive must not turn a tool call into a crawl."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_deep111"))
    deep = workdir
    for level in range(8):
        deep = deep / f"level{level}"
        deep.mkdir()
        (deep / "f.txt").write_text("x")
    found = tools._snapshot_workdir_files(str(workdir))
    assert found, "nothing was found at all"
    assert max(name.count("/") + 1 for name in found) <= tools._MAX_SANDBOX_PATH_SEGMENTS


def test_files_written_before_a_timeout_are_still_reported(tmp_path, monkeypatch):
    """`printf data > report.csv; sleep 999` produced that file."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    result = tools._bash_exec(
        "printf data > report.csv; sleep 30",
        timeout = 3,
        session_id = "__LOCALID_slowrun",
    )
    assert "timed out" in result
    assert "__FILES__:" in result, result
    assert "report.csv" in result


def test_the_image_list_is_capped_like_the_file_list(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_manyimg"))
    before = tools._snapshot_workdir_files(str(workdir))
    for i in range(80):
        (workdir / f"frame{i:03d}.png").write_bytes(b"\x89PNG")
    sentinels = tools._created_file_sentinels(str(workdir), before)
    images = json.loads(sentinels.split("__IMAGES__:")[1])
    files = json.loads(sentinels.split("__FILES__:")[1].split("\n")[0])
    assert len(images) == tools._MAX_REPORTED_FILES
    assert len(files) == tools._MAX_REPORTED_FILES


def test_a_tool_printing_the_files_marker_keeps_its_output():
    """Only a structurally valid trailing envelope is stripped."""
    from core.inference.tool_loop_controller import strip_result_for_model

    printed = "here is the doc:\n__FILES__:see the manual\nand the answer is 42"
    assert strip_result_for_model(printed) == printed

    real = 'output\n__FILES__:[{"name": "a.csv", "size": 3}]'
    assert strip_result_for_model(real) == "output"


def test_clearing_all_chats_cleans_up_their_sandboxes(tmp_path, monkeypatch):
    """The UI's "Clear all chats" is the common bulk delete."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    empty = Path(tools.get_sandbox_workdir("__LOCALID_bulk111"))
    assert empty.is_dir()

    import routes.chat_history as chat_history

    monkeypatch.setattr(chat_history, "list_chat_threads", lambda: [{"id": "__LOCALID_bulk111"}])
    monkeypatch.setattr(chat_history, "clear_chat_history", lambda: [])
    monkeypatch.setattr(chat_history, "_cancel_active_research", lambda request, ids: None)

    import asyncio

    body = asyncio.run(chat_history.clear_history(request = None, current_subject = "tester"))
    assert body["sandboxes_removed"] == 1
    assert not empty.exists()


def test_the_legacy_migration_is_startup_work(tmp_path, monkeypatch):
    """Across filesystems it copies every session, which is not something a
    listing or a download can wait on: those run on the event loop."""
    home = tmp_path / "home"
    fake_home = tmp_path / "userprofile"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    legacy = fake_home / "studio_sandbox" / "__LOCALID_upgrade"
    legacy.mkdir(parents = True)
    (legacy / "sales.csv").write_text("a,b\n")

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    # A read does not move anything.
    Path(tools.resolve_sandbox_workdir("__LOCALID_upgrade"))
    assert (legacy / "sales.csv").is_file()

    tools.migrate_legacy_sandbox_in_background()
    for _ in range(50):
        if not legacy.exists():
            break
        time.sleep(0.05)
    resolved = Path(tools.resolve_sandbox_workdir("__LOCALID_upgrade"))
    assert (resolved / "sales.csv").is_file(), "the file did not follow the migration"


def test_the_migration_is_serialised(tmp_path, monkeypatch):
    """Two chats running their first tool call at once must not strand files."""
    import threading

    fake_home = tmp_path / "userprofile"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    for index in range(6):
        session = fake_home / "studio_sandbox" / f"__LOCALID_race{index}"
        session.mkdir(parents = True)
        (session / "data.csv").write_text("a\n")

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    results = []

    def first_tool_call(index):
        results.append(Path(tools.get_sandbox_workdir(f"__LOCALID_race{index}")))

    threads = [threading.Thread(target = first_tool_call, args = (i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 6
    for workdir in results:
        assert (workdir / "data.csv").is_file(), f"{workdir} lost its files"


def test_every_reported_file_is_downloadable(tmp_path, monkeypatch):
    """The walk and the download route must agree, or the card advertises a
    file that always 404s."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from fastapi import HTTPException

    from core.inference import tools
    from routes.inference import _contained_sandbox_path

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_agree11"))
    before = tools._snapshot_workdir_files(str(workdir))
    deep = workdir
    for level in range(6):
        deep = deep / f"d{level}"
        deep.mkdir()
        (deep / f"f{level}.csv").write_text("x")

    reported = json.loads(
        tools._created_file_sentinels(str(workdir), before).split("__FILES__:")[1].split("\n")[0]
    )
    assert reported, "nothing was reported at all"
    for entry in reported:
        # Must not raise: whatever the snapshot names, the route must serve.
        try:
            _sandbox_dir, resolved = _contained_sandbox_path("__LOCALID_agree11", entry["name"])
        except HTTPException as refused:
            raise AssertionError(f"{entry['name']} reported but refused ({refused.status_code})")
        assert os.path.isfile(resolved), entry["name"]


def test_a_same_timestamp_overwrite_is_still_reported(tmp_path, monkeypatch):
    """Coarse-resolution volumes (FAT/exFAT) can repeat mtime_ns."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_coarse1"))
    target = workdir / "report.csv"
    target.write_text("a")
    before = tools._snapshot_workdir_files(str(workdir))
    target.write_text("a,b,c,d")
    os.utime(target, ns = (before["report.csv"][0], before["report.csv"][0]))  # same tick
    assert "report.csv" in tools._created_file_sentinels(str(workdir), before)


@pytest.mark.parametrize(
    "payload",
    [
        "[null]",
        "[1, 2]",
        '["a.csv"]',
        '[{"size": 3}]',
        '[{"name": ""}]',
        '[{"name": 5}]',
        '[{"name": "a.csv", "size": "big"}]',
    ],
)
def test_a_malformed_files_envelope_is_left_as_text(payload):
    """Only the executor's own envelope is protocol; anything else is output."""
    from core.inference.tool_loop_controller import strip_result_for_model

    printed = f"tool output\n__FILES__:{payload}"
    assert strip_result_for_model(printed) == printed


def test_a_dotfile_a_tool_creates_is_reported(tmp_path, monkeypatch):
    """.gitignore is a real artifact and the route serves it; only the noisy
    dot-DIRECTORIES stay out."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_dotfile"))
    before = tools._snapshot_workdir_files(str(workdir))
    (workdir / ".gitignore").write_text("*.pyc\n")
    (workdir / ".env.example").write_text("KEY=\n")
    noise = workdir / ".cache"
    noise.mkdir()
    (noise / "junk.bin").write_bytes(b"x")

    reported = tools._snapshot_workdir_files(str(workdir))
    assert ".gitignore" in reported and ".env.example" in reported
    assert not any(name.startswith(".cache/") for name in reported), reported
    assert ".gitignore" in tools._created_file_sentinels(str(workdir), before)


def test_a_name_the_route_would_refuse_is_never_reported(tmp_path, monkeypatch):
    """A backslash or control character is legal in a POSIX filename but the
    download route rejects it, so the card must not offer it."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_badname"))
    before = tools._snapshot_workdir_files(str(workdir))
    for name in ("back\\slash.csv", "bell\x07.csv", "ok.csv"):
        try:
            (workdir / name).write_text("x")
        except OSError:
            continue

    reported = tools._snapshot_workdir_files(str(workdir))
    assert "ok.csv" in reported
    assert not any("\\" in name or any(ord(c) < 32 for c in name) for name in reported), reported
    del before


def test_deleting_a_chat_right_after_an_upgrade_finds_its_legacy_sandbox(tmp_path, monkeypatch):
    """The first thing a user does may be a delete, before any tool has run."""
    fake_home = tmp_path / "userprofile"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    legacy = fake_home / "studio_sandbox" / "__LOCALID_oldchat"
    legacy.mkdir(parents = True)
    (legacy / "sales.csv").write_text("a,b\n")

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    assert tools.remove_session_sandbox("__LOCALID_oldchat", delete_files = True) is True
    assert not legacy.exists()
    assert not (Path(tools.sandbox_root()) / "__LOCALID_oldchat").exists()


def test_a_configured_cache_that_holds_other_files_is_never_deleted(tmp_path, monkeypatch):
    """UNSLOTH_COMPILE_LOCATION is user-set. Pointed at a shared directory it
    would otherwise be rmtree'd at startup, taking whatever else lives there."""
    shared = tmp_path / "shared"
    (shared / "important").mkdir(parents = True)
    (shared / "important" / "notes.txt").write_text("user data")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(shared))

    from utils import cache_cleanup

    cache_cleanup.clear_unsloth_compiled_cache()
    assert (shared / "important" / "notes.txt").read_text() == "user data"

    cache_cleanup.clear_unsloth_compiled_cache(preserve_patterns = ["Unsloth*Trainer.py"])
    assert (shared / "important" / "notes.txt").read_text() == "user data"


def test_an_unmarked_configured_cache_keeps_what_studio_did_not_write(tmp_path, monkeypatch):
    cache = tmp_path / "compiled_cache"
    cache.mkdir()
    (cache / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n")
    (cache / "__pycache__").mkdir()
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(cache))

    from utils import cache_cleanup

    cache_cleanup.clear_unsloth_compiled_cache()
    assert not (cache / "unsloth_compiled_module_gemma3.py").exists()
    assert (cache / "__pycache__").is_dir()


def test_a_symlinked_session_cannot_serve_files_outside_the_sandbox(tmp_path, monkeypatch):
    """A legacy session entry that is a symlink used to become the sandbox root
    for reads, so anything under its target was downloadable."""
    fake_home = tmp_path / "userprofile"
    fake_home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    legacy = fake_home / "studio_sandbox"
    legacy.mkdir(parents = True)
    (legacy / "__LOCALID_evil").symlink_to(outside)

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    tools._migrate_legacy_sandbox(tools.sandbox_root())
    resolved = tools.resolve_sandbox_workdir("__LOCALID_evil")
    # The link is left at the legacy root rather than carried across, so the
    # name inside the sandbox root is a plain path with nothing behind it.
    assert not (Path(tools.sandbox_root()) / "__LOCALID_evil").is_symlink()
    assert (legacy / "__LOCALID_evil").is_symlink(), "moved a link into the root"
    assert not Path(resolved).is_symlink()
    assert not str(Path(resolved).resolve()).startswith(str(outside.resolve()))
    assert not (Path(resolved) / "secret.txt").exists()


def test_the_executor_leaves_nothing_in_the_sandbox(tmp_path, monkeypatch):
    """Its scratch script lives outside the sandbox, so a chat whose tools only
    printed is still an empty folder and removable without the opt-in."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_scratch"))
    tools._python_exec("print('hi')", session_id = "__LOCALID_scratch")
    assert [p.name for p in workdir.iterdir()] == [tools._SANDBOX_MARKER]

    assert tools.remove_session_sandbox("__LOCALID_scratch") is True
    assert not workdir.exists()


def test_a_real_file_still_blocks_removal(tmp_path, monkeypatch):
    """Only scratch is ignored; the user's own files still need the opt-in."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_keepit"))
    (workdir / "studio_exec_abc123.py").write_text("print(1)")
    (workdir / "sales.csv").write_text("a,b\n")

    assert tools.remove_session_sandbox("__LOCALID_keepit") is False
    assert (workdir / "sales.csv").is_file()


def test_the_listing_drops_a_directory_the_route_would_refuse(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools
    from routes.inference import _sandbox_listing_names

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_baddir"))
    try:
        bad = workdir / "back\\slash"
        bad.mkdir()
        (bad / "report.csv").write_text("x")
    except OSError:
        pytest.skip("filesystem rejects the name")
    (workdir / "ok.csv").write_text("x")

    names = _sandbox_listing_names(str(workdir))
    assert "ok.csv" in names
    assert not any("\\" in name for name in names), names


def test_a_user_file_named_like_scratch_is_kept(tmp_path, monkeypatch):
    """No filename is reserved: everything in the sandbox is the user's."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_prefix1"))
    (workdir / "studio_exec_results.csv").write_text("a,b\n")
    (workdir / "studio_exec_ab12cd.py").write_text("print(1)")

    snapshot = tools._snapshot_workdir_files(str(workdir))
    assert "studio_exec_results.csv" in snapshot
    assert "studio_exec_ab12cd.py" in snapshot
    # Both block removal without the opt-in.
    assert tools.remove_session_sandbox("__LOCALID_prefix1") is False
    assert (workdir / "studio_exec_results.csv").is_file()
    assert (workdir / "studio_exec_ab12cd.py").is_file()


def test_an_existing_sandbox_override_keeps_its_permissions(tmp_path, monkeypatch):
    """UNSLOTH_STUDIO_SANDBOX_HOME can name a shared directory; locking it down
    to 0o700 would cut off everything else using it."""
    shared = tmp_path / "shared"
    shared.mkdir(mode = 0o755)
    before = shared.stat().st_mode & 0o777
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(shared))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_shared1"))
    assert (shared.stat().st_mode & 0o777) == before, "the shared root was re-permissioned"
    # The session directory is ours, so it is still locked down.
    assert (workdir.stat().st_mode & 0o777) == 0o700


def test_a_sandbox_root_studio_creates_is_still_locked_down(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "fresh"))

    from core.inference import tools

    tools._workdirs.clear()
    tools.get_sandbox_workdir("__LOCALID_fresh11")
    assert ((tmp_path / "fresh").stat().st_mode & 0o777) == 0o700


def test_a_directory_of_plain_python_files_is_not_a_cache(tmp_path, monkeypatch):
    """A package or scripts directory is only .py files too, so shape alone
    cannot decide what gets deleted."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "helper.py").write_text("print(1)\n")
    (scripts / "__pycache__").mkdir()
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(scripts))

    from utils import cache_cleanup

    cache_cleanup.clear_unsloth_compiled_cache()
    assert (scripts / "helper.py").read_text() == "print(1)\n"


def test_a_marked_directory_is_cleared(tmp_path, monkeypatch):
    """Studio writes the marker when it creates the location."""
    cache = tmp_path / "compiled_cache"
    cache.mkdir()
    (cache / "helper.py").write_text("print(1)\n")

    from utils import cache_cleanup

    (cache / cache_cleanup.CACHE_MARKER).touch()
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(cache))
    cache_cleanup.clear_unsloth_compiled_cache()
    assert not (cache / "helper.py").exists()
    # Emptied, not unmade: the marker is what keeps it ours next time.
    assert list(cache.iterdir()) == [cache / cache_cleanup.CACHE_MARKER]


def test_generated_modules_identify_a_cache_without_a_marker(tmp_path, monkeypatch):
    """An install that predates the marker is still cleaned, file by file."""
    cache = tmp_path / "old_cache"
    cache.mkdir()
    (cache / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n", encoding = "utf-8")
    # Their own file, and Unsloth*Trainer.py is a name a user's subclass can
    # carry: without the marker there is nothing to say we wrote it.
    (cache / "UnslothCustomTrainer.py").write_text("class X: pass\n", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(cache))

    from utils import cache_cleanup

    cache_cleanup.clear_unsloth_compiled_cache()
    assert not (cache / "unsloth_compiled_module_gemma3.py").exists()
    assert (cache / "UnslothCustomTrainer.py").is_file()


def test_studio_writes_the_marker_when_it_creates_the_location(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    from utils import cache_cleanup
    from utils.paths import storage_roots

    storage_roots.setup_cache_env()
    pinned = Path(os.environ["UNSLOTH_COMPILE_LOCATION"])
    assert (pinned / cache_cleanup.CACHE_MARKER).is_file()


def test_a_cached_sandbox_path_is_re_checked(tmp_path, monkeypatch):
    """A session directory swapped for a symlink after it was cached would
    otherwise keep serving from wherever it now points."""
    root = tmp_path / "sb"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_swapped"))
    assert workdir.is_dir()

    # Swap it, leaving the resolver's cache pointing at the same string.
    shutil.rmtree(workdir)
    workdir.symlink_to(outside)

    resolved = Path(tools.resolve_sandbox_workdir("__LOCALID_swapped"))
    assert resolved.name != "__LOCALID_swapped", resolved
    assert not (resolved / "secret.txt").exists()

    executing = Path(tools.get_sandbox_workdir("__LOCALID_swapped"))
    assert executing.name != "__LOCALID_swapped", executing
    assert not executing.is_symlink() and executing.is_dir()
    assert not (executing / "secret.txt").exists()


def test_a_shared_compile_location_loses_only_generated_files(tmp_path, monkeypatch):
    """UNSLOTH_COMPILE_LOCATION=$HOME/.cache after one compile made the whole
    directory look like ours; only the compiler's own output is."""
    shared = tmp_path / "dot_cache"
    (shared / "pip").mkdir(parents = True)
    (shared / "pip" / "wheel.whl").write_text("wheel")
    (shared / "notes.txt").write_text("keep me")
    (shared / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(shared))

    from utils import cache_cleanup

    cache_cleanup.clear_unsloth_compiled_cache()
    assert shared.is_dir()
    assert (shared / "pip" / "wheel.whl").read_text() == "wheel"
    assert (shared / "notes.txt").read_text() == "keep me"
    assert not (shared / "unsloth_compiled_module_gemma3.py").exists()


def test_a_shared_compile_location_keeps_preserved_patterns(tmp_path, monkeypatch):
    shared = tmp_path / "dot_cache2"
    shared.mkdir()
    (shared / "UnslothSFTTrainer.py").write_text("trainer")
    (shared / "unsloth_compiled_module_llama.py").write_text("x = 1\n")
    (shared / "__pycache__").mkdir()
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(shared))

    from utils import cache_cleanup

    cache_cleanup.clear_unsloth_compiled_cache(preserve_patterns = ["Unsloth*Trainer.py"])
    assert (shared / "UnslothSFTTrainer.py").is_file()
    assert not (shared / "unsloth_compiled_module_llama.py").exists()
    # Not ours to remove in a directory we do not own.
    assert (shared / "__pycache__").is_dir()


def test_a_marked_shared_directory_is_still_cleared_whole(tmp_path, monkeypatch):
    """The marker means Studio made the directory, so the old behaviour stands."""
    cache = tmp_path / "marked"
    cache.mkdir()

    from utils import cache_cleanup

    (cache / cache_cleanup.CACHE_MARKER).touch()
    (cache / "leftover.txt").write_text("x")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(cache))
    cache_cleanup.clear_unsloth_compiled_cache()
    assert not (cache / "leftover.txt").exists()
    assert list(cache.iterdir()) == [cache / cache_cleanup.CACHE_MARKER]


def test_many_empty_directories_do_not_stall_the_snapshot(tmp_path, monkeypatch):
    """Directories never hit the file cap, so the walk needs its own budget."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_manydirs"))
    monkeypatch.setattr(tools, "_MAX_SNAPSHOT_DIRS", 5)
    for i in range(40):
        (workdir / f"out{i}").mkdir()

    visited = []
    real_walk = os.walk

    def counting_walk(top, *a, **kw):
        for entry in real_walk(top, *a, **kw):
            visited.append(entry[0])
            yield entry

    monkeypatch.setattr(tools.os, "walk", counting_walk)
    tools._snapshot_workdir_files(str(workdir))
    # The budget is checked on entry, so the one that trips it is visited too.
    assert len(visited) <= 6, len(visited)


def test_a_files_marker_with_absurd_nesting_does_not_abort_the_turn(tmp_path):
    """json.loads raises RecursionError, which is not a ValueError."""
    from core.inference import tool_loop_controller

    payload = "[" * 200000 + "]" * 200000
    text = "tool output\n__FILES__:" + payload
    assert tool_loop_controller.strip_result_for_model(text) == text


def test_the_listing_walk_is_bounded_like_the_snapshot(tmp_path, monkeypatch):
    """Same directory budget: a listing request must not crawl a filesystem
    either, and it runs on a server worker."""
    from core.inference import tools
    from routes import inference

    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    for i in range(40):
        (sandbox / f"out{i}").mkdir()

    visited = []
    real_walk = os.walk

    def counting_walk(top, *a, **kw):
        for entry in real_walk(top, *a, **kw):
            visited.append(entry[0])
            yield entry

    monkeypatch.setattr(inference, "_MAX_SNAPSHOT_DIRS", 5)
    monkeypatch.setattr(inference.os, "walk", counting_walk)
    inference._sandbox_listing_names(str(sandbox))
    assert len(visited) <= 6, len(visited)
    assert inference._MAX_SNAPSHOT_FILES == tools._MAX_SNAPSHOT_FILES


def test_the_listing_shows_a_user_file_named_like_scratch(tmp_path):
    """Nothing in the sandbox belongs to the executor, so nothing is hidden."""
    from routes import inference

    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    (sandbox / "studio_exec_results.csv").write_text("a,b\n")
    (sandbox / "studio_exec_ab12cd.py").write_text("print(1)")

    names = inference._sandbox_listing_names(str(sandbox))
    assert names == ["studio_exec_ab12cd.py", "studio_exec_results.csv"], names


def test_clear_all_chats_can_take_the_files_too(tmp_path, monkeypatch):
    """DELETE /threads has the opt-in; without it here a bulk clear could only
    ever leave a nonempty sandbox behind, with no thread left to reach it."""
    import inspect

    from routes import chat_history

    signature = inspect.signature(chat_history.clear_history)
    assert "delete_files" in signature.parameters
    assert signature.parameters["delete_files"].default is False

    source = inspect.getsource(chat_history.clear_history)
    assert "_remove_sandboxes(" in source
    assert "delete_files" in source.split("_remove_sandboxes(", 1)[1]


def test_a_failed_legacy_move_is_retried(tmp_path, monkeypatch):
    """A file locked by another process on Windows is retryable; giving up after
    one attempt strands it once the destination directory exists."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox" / "__LOCALID_locked1"
    legacy.mkdir(parents = True)
    (legacy / "results.csv").write_text("a,b\n")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False

    attempts = []
    real_move = tools.shutil.move

    def failing_move(source, target):
        attempts.append(source)
        if len(attempts) == 1:
            raise OSError(13, "in use by another process")
        return real_move(source, target)

    monkeypatch.setattr(tools.shutil, "move", failing_move)
    root = tools.sandbox_root()
    tools._migrate_legacy_sandbox(root)
    assert tools._legacy_sandbox_migrated is False, "gave up on a retryable failure"

    tools._migrate_legacy_sandbox(root)
    assert tools._legacy_sandbox_migrated is True
    assert (Path(root) / "__LOCALID_locked1" / "results.csv").read_text() == "a,b\n"


def test_a_collision_is_not_a_retryable_failure(tmp_path, monkeypatch):
    """The new root already has that session, so the legacy copy stays for the
    user to find and the migration is still done."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox" / "__LOCALID_dupe123"
    legacy.mkdir(parents = True)
    (legacy / "old.csv").write_text("old\n")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    root = Path(tools.sandbox_root())
    existing = root / "__LOCALID_dupe123"
    existing.mkdir(parents = True)
    # Claimed, which is what a session directory the new root made looks like.
    (existing / tools._SANDBOX_MARKER).write_text("__LOCALID_dupe123", encoding = "utf-8")

    tools._migrate_legacy_sandbox(str(root))
    assert tools._legacy_sandbox_migrated is True
    assert (legacy / "old.csv").is_file(), "the legacy copy was not left behind"


def test_deleting_a_symlinked_session_spares_the_chat_it_points_at(tmp_path, monkeypatch):
    """realpath containment passes for a sibling, so a link left in the sandbox
    could take another chat's files."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    victim = Path(tools.get_sandbox_workdir("__LOCALID_victim1"))
    (victim / "report.csv").write_text("a,b\n")
    link = victim.parent / "__LOCALID_link111"
    link.symlink_to(victim)

    assert tools.remove_session_sandbox("__LOCALID_link111", delete_files = True) is True
    assert not link.exists(), "the stale link stayed behind"
    assert (victim / "report.csv").read_text() == "a,b\n", "the other chat lost its files"
    assert victim.is_dir()


def test_the_marker_survives_a_cache_clear(tmp_path, monkeypatch):
    """Startup clears the cache it just created, and nothing else rewrites the
    marker, so without this our own cache is demoted to 'shared'."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    from utils import cache_cleanup
    from utils.paths import storage_roots

    storage_roots.setup_cache_env()
    pinned = Path(os.environ["UNSLOTH_COMPILE_LOCATION"])
    (pinned / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n", encoding = "utf-8")

    cache_cleanup.clear_unsloth_compiled_cache()
    assert (pinned / cache_cleanup.CACHE_MARKER).is_file()
    assert not (pinned / "unsloth_compiled_module_gemma3.py").exists()

    # Still ours on the next pass, so a __pycache__ left by the compiler goes too.
    (pinned / "__pycache__").mkdir()
    (pinned / "UnslothSFTTrainer.py").write_text("trainer\n", encoding = "utf-8")
    cache_cleanup.clear_unsloth_compiled_cache(preserve_patterns = ["Unsloth*Trainer.py"])
    assert not (pinned / "__pycache__").exists()
    assert (pinned / "UnslothSFTTrainer.py").is_file()
    assert (pinned / cache_cleanup.CACHE_MARKER).is_file()


def test_an_unrelated_cache_named_folder_in_the_cwd_is_not_ours(tmp_path, monkeypatch):
    """Studio is launched from wherever the shell happens to be, so the name
    alone cannot license an rmtree."""
    launch_dir = tmp_path / "someproject"
    cache = launch_dir / "unsloth_compiled_cache"
    cache.mkdir(parents = True)
    (cache / "notes.txt").write_text("keep me")
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    from utils import cache_cleanup

    cache_cleanup.clear_unsloth_compiled_cache()
    assert (cache / "notes.txt").read_text() == "keep me"


def test_a_marked_cwd_cache_is_still_cleared(tmp_path, monkeypatch):
    launch_dir = tmp_path / "studioproject"
    cache = launch_dir / "unsloth_compiled_cache"
    cache.mkdir(parents = True)
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    from utils import cache_cleanup

    (cache / cache_cleanup.CACHE_MARKER).touch()
    (cache / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n")
    cache_cleanup.clear_unsloth_compiled_cache()
    assert not (cache / "unsloth_compiled_module_gemma3.py").exists()


def test_a_user_python_file_is_never_executor_scratch(tmp_path, monkeypatch):
    """The executor's own file no longer lives in the sandbox, so no filename
    is reserved and studio_exec_results.py is just a file."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools
    from routes import inference

    tools._workdirs.clear()
    session = "__LOCALID_userpy1"
    result = tools._python_exec(
        "open('studio_exec_results.py','w').write('x = 1\\n')", session_id = session
    )
    assert "studio_exec_results.py" in result, result

    workdir = Path(tools.get_sandbox_workdir(session))
    # The executor left nothing of its own behind.
    assert sorted(
        p.name for p in workdir.iterdir() if p.name not in tools._INTERNAL_SANDBOX_FILES
    ) == ["studio_exec_results.py"]
    assert inference._sandbox_listing_names(str(workdir)) == ["studio_exec_results.py"]
    # And a delete without the opt-in will not quietly take it.
    assert tools.remove_session_sandbox(session) is False
    assert (workdir / "studio_exec_results.py").is_file()


def test_a_sandbox_root_at_a_filesystem_root_still_isolates_chats(tmp_path, monkeypatch):
    """A root already ending in a separator made every session path fail
    containment, so every chat collapsed into _invalid together."""
    from core.inference import tools

    assert tools._contained_in_root("/srv/chat", "/") is True
    assert tools._contained_in_root("/", "/") is False
    assert tools._contained_in_root("/etc/passwd", str(tmp_path)) is False

    root = tmp_path / "sb"
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))
    tools._workdirs.clear()
    first = Path(tools.get_sandbox_workdir("__LOCALID_alpha1"))
    second = Path(tools.get_sandbox_workdir("__LOCALID_beta22"))
    assert first != second
    assert first.name != "_invalid" and second.name != "_invalid"


def test_containment_is_not_fooled_by_a_shared_name_prefix(tmp_path):
    """The old prefix test would also have accepted a sibling like sb-evil."""
    from core.inference import tools

    (tmp_path / "sb").mkdir()
    (tmp_path / "sb-evil").mkdir()
    assert tools._contained_in_root(str(tmp_path / "sb-evil"), str(tmp_path / "sb")) is False
    assert tools._contained_in_root(str(tmp_path / "sb" / "chat"), str(tmp_path / "sb")) is True


def test_studios_own_sandbox_bookkeeping_is_not_a_user_file(tmp_path, monkeypatch):
    """The remap sidecar the sandbox sitecustomize writes is ours, and reporting
    it also made a streamed result differ from a non-streamed one."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools
    from routes import inference

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_remap01"))
    (workdir / ".unsloth_sandbox_remap.json").write_text("{}")
    (workdir / ".gitignore").write_text("*.pyc\n")

    snapshot = tools._snapshot_workdir_files(str(workdir))
    assert ".unsloth_sandbox_remap.json" not in snapshot
    # Other dotfiles are still the user's.
    assert ".gitignore" in snapshot
    assert inference._sandbox_listing_names(str(workdir)) == [".gitignore"]


def test_a_module_written_by_an_earlier_call_is_importable(tmp_path, monkeypatch):
    """The scratch script is what Python puts on sys.path[0], so moving it out
    of the sandbox broke `import helper` and sent __file__ outside."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_import1"
    tools._python_exec('open("helper.py", "w").write("VALUE = 42")', session_id = session)
    result = tools._python_exec("import helper; print(helper.VALUE)", session_id = session)
    assert "42" in result, result

    where = tools._python_exec("print(__file__)", session_id = session)
    workdir = tools.get_sandbox_workdir(session)
    assert workdir in where, where


def test_the_scratch_script_is_never_reported_as_a_file(tmp_path, monkeypatch):
    """Excluded by its exact name for this one call, so a tool writing
    studio_exec_results.py still keeps it."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_scratch2"
    result = tools._python_exec(
        'open("studio_exec_results.py", "w").write("x = 1")', session_id = session
    )
    files = result.split("__FILES__:")[1]
    assert "studio_exec_results.py" in files
    workdir = Path(tools.get_sandbox_workdir(session))
    # Only the user's file is left; the executor cleaned up after itself.
    assert sorted(
        p.name for p in workdir.iterdir() if p.name not in tools._INTERNAL_SANDBOX_FILES
    ) == ["studio_exec_results.py"]
    assert json.loads(files) == [{"name": "studio_exec_results.py", "size": 5}]


def test_a_turn_without_a_chat_id_reports_no_files(tmp_path, monkeypatch):
    """Every such turn shares the _default workdir, so a card pinned to it
    would later download whatever the next new chat wrote there."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    result = tools._python_exec('open("first.csv", "w").write("a")')
    assert "__FILES__" not in result, result
    assert "__IMAGES__" not in result
    bash_result = tools._bash_exec("printf a > second.csv")
    assert "__FILES__" not in bash_result, bash_result


def test_an_unowned_cwd_cache_is_not_put_on_the_import_path(tmp_path, monkeypatch):
    """Registering it would shadow real dependencies for every spawned worker."""
    launch_dir = tmp_path / "someproject"
    cache = launch_dir / "unsloth_compiled_cache"
    cache.mkdir(parents = True)
    (cache / "numpy.py").write_text("raise SystemExit('shadowed')\n")
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)
    monkeypatch.setenv("PYTHONPATH", "")

    from utils import cache_cleanup

    before = list(sys.path)
    try:
        cache_cleanup.register_compiled_cache_on_path()
        assert str(cache.resolve()) not in sys.path
        assert str(cache.resolve()) not in os.environ.get("PYTHONPATH", "")
    finally:
        sys.path[:] = before


def test_a_marked_cwd_cache_is_still_registered(tmp_path, monkeypatch):
    launch_dir = tmp_path / "studioproject"
    cache = launch_dir / "unsloth_compiled_cache"
    cache.mkdir(parents = True)
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    from utils import cache_cleanup

    (cache / cache_cleanup.CACHE_MARKER).touch()
    before = list(sys.path)
    try:
        cache_cleanup.register_compiled_cache_on_path()
        assert str(cache.resolve()) in sys.path
    finally:
        sys.path[:] = before


def test_a_chat_deleted_mid_call_keeps_its_sandbox(tmp_path, monkeypatch):
    """The running tool has this directory as its cwd, and a process whose cwd
    was unlinked fails every relative write with ENOENT."""
    import threading

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_busy111"
    workdir = Path(tools.get_sandbox_workdir(session))
    started, may_finish = threading.Event(), threading.Event()
    removed = {}

    def run_tool():
        with tools._session_in_flight(session):
            started.set()
            may_finish.wait(timeout = 10)

    worker = threading.Thread(target = run_tool)
    worker.start()
    try:
        assert started.wait(timeout = 10)
        removed["during"] = tools.remove_session_sandbox(session)
        assert removed["during"] is False, "the sandbox went out from under a running tool"
        assert workdir.is_dir()
    finally:
        may_finish.set()
        worker.join(timeout = 10)

    # The refused request was queued, so leaving the call performs it.
    assert not workdir.exists()


def test_the_executor_marks_its_session_busy():
    """The guard has to sit on the dispatch both executors go through."""
    import inspect

    from core.inference import tools

    source = inspect.getsource(tools.execute_tool)
    for call in ("_python_exec(", "_bash_exec("):
        before = source.split(call)[0]
        assert "_session_in_flight" in before.rsplit("if name ==", 1)[-1], call


def test_a_pre_existing_compile_directory_is_never_marked_as_ours(tmp_path, monkeypatch):
    """mkdir(exist_ok=True) does not mean we created it, and the marker is
    permission to rmtree."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    from utils import cache_cleanup
    from utils.paths import storage_roots

    # Where Studio would pin it, already there and holding someone else's files.
    pinned = Path(storage_roots.cache_root()).parent / "compiled_cache"
    pinned.mkdir(parents = True)
    (pinned / "someones_notes.txt").write_text("keep me")

    storage_roots.setup_cache_env()
    assert Path(os.environ["UNSLOTH_COMPILE_LOCATION"]) == pinned
    assert not (pinned / cache_cleanup.CACHE_MARKER).exists(), "claimed a directory it found"

    cache_cleanup.clear_unsloth_compiled_cache()
    assert (pinned / "someones_notes.txt").read_text() == "keep me"


def test_a_directory_studio_creates_is_marked(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "fresh"))
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    from utils import cache_cleanup
    from utils.paths import storage_roots

    storage_roots.setup_cache_env()
    pinned = Path(os.environ["UNSLOTH_COMPILE_LOCATION"])
    assert (pinned / cache_cleanup.CACHE_MARKER).is_file()


def test_removal_and_the_busy_check_are_one_decision(tmp_path, monkeypatch):
    """Checking first and deleting after leaves a window for a tool to start in
    between, and it would then be running in a directory this call removes."""
    import threading
    import time

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_race111"
    workdir = Path(tools.get_sandbox_workdir(session))
    started = threading.Event()
    entered = threading.Event()

    real_rmtree = tools.shutil.rmtree

    def slow_rmtree(path, **kwargs):
        # Stand in for the window between deciding and unlinking.
        entered.set()
        time.sleep(0.3)
        return real_rmtree(path, **kwargs)

    monkeypatch.setattr(tools.shutil, "rmtree", slow_rmtree)
    result = {}

    def remover():
        result["removed"] = tools.remove_session_sandbox(session)

    def tool():
        entered.wait(timeout = 5)
        with tools._session_in_flight(session):
            started.set()
            time.sleep(0.05)

    threads = [threading.Thread(target = remover), threading.Thread(target = tool)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 10)

    # The tool could only take the lock after the removal finished, so it never
    # ran inside a directory that was being deleted.
    assert started.is_set()
    assert result["removed"] is True
    assert tools._active_sessions == {}


def test_the_delete_dialog_offers_the_same_choice_for_a_chat():
    """Without it every deleted chat that wrote a file leaves its sandbox
    behind with nothing left to reach it."""
    root = Path(__file__).resolve().parents[2] / "frontend" / "src"
    sidebar = (root / "components" / "app-sidebar.tsx").read_text(encoding = "utf-8")
    assert "shouldDeleteChatFiles" in sidebar
    assert "deleteChatWithCleanup(target.item, {" in sidebar

    api = (root / "features" / "chat" / "api" / "chat-api.ts").read_text(encoding = "utf-8")
    assert "delete_files: !!args.deleteFiles" in api


def test_a_delete_during_a_call_happens_once_the_call_ends(tmp_path, monkeypatch):
    """The thread is gone from history by then, so a dropped request would
    strand the folder for good."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    tools._pending_removals.clear()
    session = "__LOCALID_defer11"
    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "report.csv").write_text("a,b\n")

    with tools._session_in_flight(session):
        assert tools.remove_session_sandbox(session, delete_files = True) is False
        assert workdir.is_dir(), "removed under a running tool"
    # Queued, so leaving the call performs it.
    assert not workdir.exists()
    assert tools._pending_removals == {}


def test_a_queued_delete_keeps_the_strongest_request(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    tools._pending_removals.clear()
    session = "__LOCALID_defer22"
    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "report.csv").write_text("a,b\n")

    with tools._session_in_flight(session):
        tools.remove_session_sandbox(session, delete_files = False)
        tools.remove_session_sandbox(session, delete_files = True)
    assert not workdir.exists()


def test_a_symlinked_cache_location_is_still_cleared(tmp_path, monkeypatch):
    """rmtree refuses a symlink, and ignore_errors made that silent."""
    real = tmp_path / "real_cache"
    real.mkdir()
    link = tmp_path / "link_cache"
    link.symlink_to(real, target_is_directory = True)

    from utils import cache_cleanup

    (real / cache_cleanup.CACHE_MARKER).touch()
    (real / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(link))

    cache_cleanup.clear_unsloth_compiled_cache()
    assert not (real / "unsloth_compiled_module_gemma3.py").exists()
    assert (real / cache_cleanup.CACHE_MARKER).is_file(), "still ours next time"


def test_the_delete_switch_only_appears_where_it_works():
    """A training run has no sandbox of its own; a chat always does."""
    sidebar = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "components" / "app-sidebar.tsx"
    ).read_text(encoding = "utf-8")
    assert "function deleteTargetHasFiles" in sidebar
    assert '"training"' not in sidebar.split("function deleteTargetHasFiles")[1][:400]
    assert "{deleteTargetHasFiles(confirmingDelete) ? (" in sidebar
    # And every opener clears the switch, so it can never arrive preselected.
    assert "function openDeleteDialog" in sidebar
    assert "setConfirmingDelete({ kind:" not in sidebar


def test_a_symlinked_builtin_cache_is_not_deleted_through(tmp_path, monkeypatch):
    """Being at a built-in path proves ownership of the directory, not of
    whatever a link there points at."""
    victim = tmp_path / "someones_files"
    victim.mkdir()
    (victim / "thesis.txt").write_text("years of work")

    from utils import cache_cleanup

    link = tmp_path / "unsloth_compiled_cache"
    link.symlink_to(victim, target_is_directory = True)
    monkeypatch.setattr(cache_cleanup, "_CACHE_DIRS", [link])
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)
    monkeypatch.chdir(tmp_path / "..")

    cache_cleanup.clear_unsloth_compiled_cache()
    assert (victim / "thesis.txt").read_text() == "years of work"


def test_a_symlinked_builtin_cache_with_the_marker_is_cleared(tmp_path, monkeypatch):
    real = tmp_path / "real_cache"
    real.mkdir()

    from utils import cache_cleanup

    (real / cache_cleanup.CACHE_MARKER).touch()
    (real / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n")
    link = tmp_path / "unsloth_compiled_cache"
    link.symlink_to(real, target_is_directory = True)
    monkeypatch.setattr(cache_cleanup, "_CACHE_DIRS", [link])
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    cache_cleanup.clear_unsloth_compiled_cache()
    assert not (real / "unsloth_compiled_module_gemma3.py").exists()


def test_the_invalid_fallback_cannot_be_pointed_out_of_the_sandbox(tmp_path, monkeypatch):
    """Tool code runs inside the sandbox, so it can plant this link; every
    unchecked request would then read from wherever it points."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "id_rsa").write_text("PRIVATE KEY")
    root = tmp_path / "sb"
    root.mkdir()
    (root / "_invalid").symlink_to(outside, target_is_directory = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    resolved = Path(tools.resolve_sandbox_workdir("../escape"))
    assert not (resolved / "id_rsa").exists(), resolved
    assert outside.is_dir() and (outside / "id_rsa").is_file(), "the target was touched"

    executing = Path(tools.get_sandbox_workdir("../escape"))
    assert not (executing / "id_rsa").exists(), executing


def test_the_default_fallback_is_contained_too(tmp_path, monkeypatch):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET")
    root = tmp_path / "sb"
    root.mkdir()
    (root / "_default").symlink_to(outside, target_is_directory = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    resolved = Path(tools.resolve_sandbox_workdir(None))
    assert not (resolved / "secret.txt").exists(), resolved


def test_deleting_a_big_sandbox_does_not_hold_the_tool_lock(tmp_path, monkeypatch):
    """rmtree of a large tree would otherwise block every tool start, and the
    event loop of the route that called it."""
    import threading
    import time

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_bigdel1"
    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "data.bin").write_bytes(b"x" * 1024)

    slow = threading.Event()
    real_rmtree = tools.shutil.rmtree

    def slow_rmtree(path, **kw):
        slow.set()
        time.sleep(0.5)
        return real_rmtree(path, **kw)

    monkeypatch.setattr(tools.shutil, "rmtree", slow_rmtree)
    assert tools.remove_session_sandbox(session, delete_files = True) is True
    assert not workdir.exists(), "the session directory is gone immediately"

    # The lock is free while the tree is still being removed.
    assert slow.wait(timeout = 5)
    started = time.monotonic()
    with tools._session_in_flight("__LOCALID_other12"):
        pass
    assert time.monotonic() - started < 0.3, "a tool start waited for the delete"


def test_a_chat_whose_id_looks_like_a_project_is_still_cleaned(tmp_path, monkeypatch):
    """An imported chat can carry that prefix without a project behind it, and
    _get_workdir gives it an ordinary directory."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    monkeypatch.setattr(tools, "_get_project_workdir", lambda session_id: None)
    session = f"{tools._PROJECT_SESSION_PREFIX}notreal123"
    workdir = Path(tools.get_sandbox_workdir(session))
    assert workdir.is_dir()

    assert tools.remove_session_sandbox(session) is True
    assert not workdir.exists()


def test_a_real_project_workspace_is_still_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    shared = tmp_path / "project_workspace"
    shared.mkdir()
    monkeypatch.setattr(tools, "_get_project_workdir", lambda session_id: str(shared))
    session = f"{tools._PROJECT_SESSION_PREFIX}real123"
    assert tools.remove_session_sandbox(session, delete_files = True) is False
    assert shared.is_dir(), "a shared project workspace was removed"


def test_a_foreign_tool_result_keeps_its_own_fields():
    """Studio's wrapper always carries images; anything else with text and
    sessionId is someone else's result and must not be reduced to its text."""
    adapter = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "src"
        / "features"
        / "chat"
        / "api"
        / "chat-adapter.ts"
    ).read_text(encoding = "utf-8")
    predicate = adapter.split("export function isSandboxToolResult(", 1)[1].split("\n}", 1)[0]
    assert "Array.isArray(v.images)" in predicate, predicate


def test_a_shared_roots_own_folder_is_never_deleted(tmp_path, monkeypatch):
    """An id can name something already in a shared root, and that folder was
    not created by us."""
    root = tmp_path / "shared"
    root.mkdir()
    theirs = root / "invoices"
    theirs.mkdir()
    (theirs / "2026.pdf").write_text("money", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    assert tools.remove_session_sandbox("invoices", delete_files = True) is False
    assert (theirs / "2026.pdf").is_file()

    # Empty ones are not ours to reclaim either.
    (root / "empty").mkdir()
    assert tools.remove_session_sandbox("empty") is False
    assert (root / "empty").is_dir()


def test_a_sandbox_we_created_in_a_shared_root_is_still_removable(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_mine11"))
    assert (workdir / tools._SANDBOX_MARKER).is_file()
    # Nothing but our marker in it, so it counts as empty.
    assert tools.remove_session_sandbox("__LOCALID_mine11") is True
    assert not workdir.exists()

    workdir = Path(tools.get_sandbox_workdir("__LOCALID_mine22"))
    (workdir / "report.csv").write_text("a,b\n", encoding = "utf-8")
    assert tools.remove_session_sandbox("__LOCALID_mine22", delete_files = True) is True


def test_two_ids_differing_only_in_case_share_the_busy_check(tmp_path, monkeypatch):
    """They are one directory on Windows and on a default macOS volume, so a
    delete of one must not land while the other is running a tool in it."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    tools._pending_removals.clear()
    workdir = Path(tools.get_sandbox_workdir("ChatCase"))
    (workdir / "report.csv").write_text("a,b\n", encoding = "utf-8")

    with tools._session_in_flight("chatcase"):
        assert tools.remove_session_sandbox("ChatCase", delete_files = True) is False
        assert workdir.is_dir(), "removed under a running tool"
    # And the queued delete names the id that was asked for, not the folded key.
    assert not workdir.exists()
    assert tools._pending_removals == {}


def test_an_existing_folder_in_a_shared_root_is_never_claimed(tmp_path, monkeypatch):
    """A chat id can name something already in there, and running a tool in it
    must not make it ours to delete."""
    root = tmp_path / "shared"
    root.mkdir()
    theirs = root / "taxes"
    theirs.mkdir()
    (theirs / "2026.pdf").write_text("money", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("taxes"))
    assert workdir != theirs, "ran the tool inside a folder we did not create"
    assert workdir.name.startswith("taxes-")
    assert not (theirs / tools._SANDBOX_MARKER).exists(), "claimed a folder we did not create"
    # Deleting the chat takes our directory and leaves theirs alone.
    assert tools.remove_session_sandbox("taxes", delete_files = True) is True
    assert (theirs / "2026.pdf").is_file()


def test_both_case_variants_are_removed_when_their_calls_end(tmp_path, monkeypatch):
    """They fold onto one key, and on a case-sensitive filesystem they are two
    directories, so neither may be dropped."""
    import time

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    tools._pending_removals.clear()
    lower = Path(tools.get_sandbox_workdir("casepair"))
    upper = Path(tools.get_sandbox_workdir("CasePair"))
    if lower == upper:
        pytest.skip("case-insensitive filesystem: one directory, nothing to strand")
    (lower / "a.csv").write_text("a\n", encoding = "utf-8")
    (upper / "b.csv").write_text("b\n", encoding = "utf-8")

    with tools._session_in_flight("casepair"), tools._session_in_flight("CasePair"):
        assert tools.remove_session_sandbox("casepair", delete_files = True) is False
        assert tools.remove_session_sandbox("CasePair", delete_files = True) is False
    for _ in range(50):
        if not lower.exists() and not upper.exists():
            break
        time.sleep(0.05)
    assert not lower.exists() and not upper.exists()
    assert tools._pending_removals == {}


def test_a_foreign_fallback_link_is_left_where_it_stands(tmp_path, monkeypatch):
    """At a root the user pointed us at, _default is their entry: it must not be
    followed, and it is not ours to unlink either."""
    root = tmp_path / "shared"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET", encoding = "utf-8")
    (root / "_default").symlink_to(outside, target_is_directory = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    resolved = Path(tools.resolve_sandbox_workdir(None))
    assert not (resolved / "secret.txt").exists(), resolved
    assert (root / "_default").is_symlink(), "an entry we do not own was unlinked"

    workdir = Path(tools.get_sandbox_workdir(None))
    assert not (workdir / "secret.txt").exists(), workdir
    assert (root / "_default").is_symlink()


def test_our_own_fallback_link_is_still_dropped(tmp_path, monkeypatch):
    """At our own root nothing else put it there, so the link goes and the
    fallback keeps its name."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    root = Path(tools.sandbox_root())
    root.mkdir(parents = True, exist_ok = True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (root / "_default").symlink_to(outside, target_is_directory = True)

    workdir = Path(tools.get_sandbox_workdir(None))
    assert workdir == root / "_default"
    assert workdir.is_dir() and not workdir.is_symlink()


def test_a_pre_existing_folder_keeps_its_permissions(tmp_path, monkeypatch):
    """Running a tool in a shared root must not re-permission someone's folder."""
    if os.name == "nt":
        pytest.skip("POSIX permission bits")
    root = tmp_path / "shared"
    root.mkdir()
    theirs = root / "team"
    theirs.mkdir()
    os.chmod(theirs, 0o755)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    assert Path(tools.get_sandbox_workdir("team")) != theirs
    assert oct(theirs.stat().st_mode)[-3:] == "755", "an unowned folder was locked down"

    # Ours is still tightened.
    mine = Path(tools.get_sandbox_workdir("__LOCALID_mine33"))
    assert oct(mine.stat().st_mode)[-3:] == "700"


def test_a_link_in_a_shared_root_is_left_alone_by_a_delete(tmp_path, monkeypatch):
    """A chat id can name an entry the user put there, and a delete is not a
    licence to remove it."""
    root = tmp_path / "shared"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "notes").symlink_to(elsewhere, target_is_directory = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    assert tools.remove_session_sandbox("notes", delete_files = True) is False
    assert (root / "notes").is_symlink(), "an entry we did not create was unlinked"
    assert elsewhere.is_dir()


def test_a_case_variant_chat_gets_its_own_directory(tmp_path, monkeypatch):
    """One name on Windows and on a default macOS volume: sharing it means
    either chat's delete takes the other's files."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    first = Path(tools.get_sandbox_workdir("CaseOwn"))
    (first / "report.csv").write_text("a,b\n", encoding = "utf-8")
    assert (first / tools._SANDBOX_MARKER).read_text(encoding = "utf-8") == "CaseOwn"

    # What the other id sees on a case-insensitive volume: this directory, made
    # by someone else. On a case-sensitive one it is a separate name already.
    root = first.parent
    collision = root / "caseown"
    if not collision.exists():
        collision.mkdir()
        (collision / tools._SANDBOX_MARKER).write_text("CaseOwn", encoding = "utf-8")
        (collision / "report.csv").write_text("a,b\n", encoding = "utf-8")

    second = Path(tools._session_dir(str(root), "caseown"))
    assert second.name == "caseown-" + hashlib.sha256(b"caseown").hexdigest()[:8]

    # And that id cannot delete the other chat's files.
    assert tools.remove_session_sandbox("caseown", delete_files = True) is False
    assert (collision / "report.csv").is_file()
    assert tools.remove_session_sandbox("CaseOwn", delete_files = True) is True


def test_the_delete_switch_does_not_promise_project_files():
    """A chat moved back to Recents wrote its earlier files into the project
    workspace, which chat deletion does not touch."""
    sidebar = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "components" / "app-sidebar.tsx"
    ).read_text(encoding = "utf-8")
    assert "This chat's own sandbox folder is removed from disk." in sidebar
    assert "Anything this chat's tools wrote is removed from disk." not in sidebar


def test_a_tool_cannot_forge_its_way_into_owning_a_folder(tmp_path, monkeypatch):
    """The marker is writable by whatever runs in the sandbox, so the answer is
    to never run in a folder that was already there."""
    root = tmp_path / "shared"
    import time

    root.mkdir()
    theirs = root / "photos"
    theirs.mkdir()
    (theirs / "wedding.jpg").write_text("jpeg", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("photos"))
    assert workdir != theirs, "ran the tool inside a folder we did not create"
    assert not (theirs / tools._SANDBOX_MARKER).exists()
    # What the tool writes lands in ours, and deleting the chat takes ours.
    (workdir / "plot.png").write_text("png", encoding = "utf-8")
    assert tools.remove_session_sandbox("photos", delete_files = True) is True
    for _ in range(50):
        if not workdir.exists():
            break
        time.sleep(0.05)
    assert not workdir.exists()
    assert (theirs / "wedding.jpg").is_file()


def test_two_ids_racing_for_one_name_do_not_share_it(tmp_path, monkeypatch):
    """Both can see an unowned name before either writes its marker."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    root = tools.sandbox_root()
    os.makedirs(root, exist_ok = True)
    first = Path(tools._ensure_session_dir(root, "RaceId"))
    # The other id, resolving the same plain name (what a case-insensitive
    # volume produces): the claim is already taken, so it steps aside.
    plain = Path(root) / "raceid"
    if not plain.exists():
        plain.mkdir()
        (plain / tools._SANDBOX_MARKER).write_text("RaceId", encoding = "utf-8")
    second = Path(tools._ensure_session_dir(root, "raceid"))
    assert second != first and second != plain
    assert (second / tools._SANDBOX_MARKER).read_text(encoding = "utf-8") == "raceid"


def test_a_migrated_sandbox_stays_deletable_in_an_overridden_root(tmp_path, monkeypatch):
    """It came from our own folder, so the move has to say so or the chat can
    never remove it again."""
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    legacy = home / "studio_sandbox" / "__LOCALID_moved11"
    legacy.mkdir(parents = True)
    (legacy / "notes.txt").write_text("mine", encoding = "utf-8")
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    tools._migrate_legacy_sandbox(str(root))

    moved = root / "__LOCALID_moved11"
    assert (moved / "notes.txt").is_file()
    assert (moved / tools._SANDBOX_MARKER).is_file(), "the migrated sandbox lost its claim"
    assert tools.remove_session_sandbox("__LOCALID_moved11", delete_files = True) is True


def test_a_name_the_download_url_cannot_carry_is_not_advertised():
    """A non-UTF-8 byte in a POSIX filename arrives as a lone surrogate, and
    encodeURIComponent throws on it, so the chip could never download."""
    from core.inference import tools

    assert tools._servable_segment("report.csv")
    assert not tools._servable_segment("bad\udcffname.csv")
    assert not tools._servable_segment("\ud800.txt")


def test_a_tool_writing_over_the_marker_does_not_lose_its_files(tmp_path, monkeypatch):
    """The file sits in a directory the tool can write, so a restart must not
    send that chat somewhere else and strand what it made."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_clob11"))
    (workdir / "results.csv").write_text("a,b\n", encoding = "utf-8")
    (workdir / tools._SANDBOX_MARKER).write_text("Traceback: not an id\n", encoding = "utf-8")

    # What the next launch sees.
    tools._workdirs.clear()
    again = Path(tools.get_sandbox_workdir("__LOCALID_clob11"))
    assert again == workdir, "the chat was sent to a new directory"
    assert (again / "results.csv").is_file()
    # And the claim is back, so deletion still works.
    assert (again / tools._SANDBOX_MARKER).read_text(encoding = "utf-8") == "__LOCALID_clob11"
    assert Path(tools.resolve_sandbox_workdir("__LOCALID_clob11")) == workdir


def test_a_legacy_name_taken_in_a_shared_root_moves_beside_it(tmp_path, monkeypatch):
    """The name is somebody's own folder, so the session moves to the name it
    will resolve to instead of merging into theirs."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox" / "notes"
    legacy.mkdir(parents = True)
    (legacy / "mine.csv").write_text("mine", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    root = tmp_path / "shared"
    root.mkdir()
    theirs = root / "notes"
    theirs.mkdir()
    (theirs / "theirs.txt").write_text("theirs", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    tools._migrate_legacy_sandbox(str(root))

    assert (theirs / "theirs.txt").is_file()
    assert not (theirs / "mine.csv").exists(), "merged into a folder we do not own"
    moved = Path(tools.get_sandbox_workdir("notes"))
    assert moved != theirs
    assert (moved / "mine.csv").read_text(encoding = "utf-8") == "mine"


def test_an_empty_compile_location_is_not_an_override(tmp_path, monkeypatch):
    """An inherited KEY= would pin the cache to "", which puts an empty entry on
    sys.path and sends the compiler to the system temp directory."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", "")

    from utils.paths import storage_roots

    storage_roots.setup_cache_env()
    pinned = os.environ["UNSLOTH_COMPILE_LOCATION"]
    assert pinned.strip(), "the cache was left unpinned"
    assert Path(pinned).is_dir()


def test_an_id_the_filesystem_cannot_hold_still_gets_its_own_directory(tmp_path, monkeypatch):
    """These come from API clients. One shared bucket meant every such chat
    could read, and delete, every other one's files."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    first = Path(tools.get_sandbox_workdir("chat.one"))
    second = Path(tools.get_sandbox_workdir("chat.two"))
    assert first != second
    assert first.name.startswith("_id-") and second.name.startswith("_id-")
    (first / "mine.csv").write_text("mine", encoding = "utf-8")

    # Stable across a restart, so a download chip still resolves.
    tools._workdirs.clear()
    assert Path(tools.resolve_sandbox_workdir("chat.one")) == first
    # And nothing traverses: the name is derived, not the id.
    escape = Path(tools.get_sandbox_workdir("../../etc"))
    assert escape.parent == Path(tools.sandbox_root())


def test_a_foreign_folder_is_not_taken_for_an_interrupted_move(tmp_path, monkeypatch):
    """Entry names overlapping is not provenance: only the record written
    before a move says the directory was ours to fill."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox" / "shared_name"
    legacy.mkdir(parents = True)
    (legacy / "notes.txt").write_text("mine", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    root = tmp_path / "shared"
    root.mkdir()
    theirs = root / "shared_name"
    theirs.mkdir()
    # A subset of the legacy names, which used to be enough to look partial.
    (theirs / "notes.txt").write_text("theirs", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    tools._migrate_legacy_sandbox(str(root))

    assert not (theirs / tools._SANDBOX_MARKER).exists(), "claimed a folder we never made"
    assert (theirs / "notes.txt").read_text(encoding = "utf-8") == "theirs"


def test_a_literal_id_cannot_take_a_derived_name(tmp_path, monkeypatch):
    """_id-<hash> is the name an unusable id resolves to, so a chat called that
    must not land on the same directory."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    derived = tools._sandbox_name("chat.one")
    assert derived.startswith(tools._DERIVED_PREFIX)
    assert tools._sandbox_name(derived) != derived, "a literal id took a derived name"

    first = Path(tools.get_sandbox_workdir("chat.one"))
    (first / "mine.csv").write_text("mine", encoding = "utf-8")
    second = Path(tools.get_sandbox_workdir(derived))
    assert second != first
    assert not (second / "mine.csv").exists()


def test_a_link_inside_the_root_is_stepped_around(tmp_path, monkeypatch):
    """Its target is contained, so containment alone accepts it, and claiming
    through it writes our marker into a directory we never made."""
    root = tmp_path / "shared"
    root.mkdir()
    foreign = root / "foreign"
    foreign.mkdir()
    (foreign / "theirs.txt").write_text("theirs", encoding = "utf-8")
    (root / "chat").symlink_to(foreign, target_is_directory = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("chat"))
    assert workdir.resolve() != foreign.resolve(), workdir
    assert not (foreign / tools._SANDBOX_MARKER).exists(), "claimed through a link"
    assert (foreign / "theirs.txt").is_file()


def test_an_unowned_cache_of_trainers_is_not_put_on_sys_path(tmp_path, monkeypatch):
    """Unsloth*Trainer.py is a name a user's own subclass carries, and anything
    else in that directory would then shadow real modules for every worker."""
    import sys as _sys

    theirs = tmp_path / "unsloth_compiled_cache"
    theirs.mkdir()
    (theirs / "UnslothCustomTrainer.py").write_text("class X: pass\n", encoding = "utf-8")
    (theirs / "numpy.py").write_text("raise SystemExit('shadowed')\n", encoding = "utf-8")
    # In the launch directory, which is the case that is not ours.
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "")

    from utils import cache_cleanup

    before = list(_sys.path)
    cache_cleanup.register_compiled_cache_on_path()
    assert str(theirs.resolve()) not in _sys.path, "an unowned directory went on sys.path"
    assert str(theirs.resolve()) not in os.environ.get("PYTHONPATH", "")
    _sys.path[:] = before

    # One the compiler has actually written into is still registered.
    (theirs / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n", encoding = "utf-8")
    cache_cleanup.register_compiled_cache_on_path()
    assert str(theirs.resolve()) in _sys.path
    _sys.path[:] = before


def test_the_delete_switch_reaches_a_chat_moved_into_a_project():
    """Anything it wrote before the move is in its own folder, and the backend
    never touches the project workspace."""
    sidebar = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "components" / "app-sidebar.tsx"
    ).read_text(encoding = "utf-8")
    assert 'return target.kind === "project" || target.kind === "chat";' in sidebar
    assert "!target.item.projectId" not in sidebar


def test_a_persisted_files_value_that_is_not_a_list_is_not_a_wrapper():
    """The cards map over it, so anything else takes the chat view down."""
    root = Path(__file__).resolve().parents[2] / "frontend" / "src"
    adapter = (root / "features" / "chat" / "api" / "chat-adapter.ts").read_text(encoding = "utf-8")
    python_card = (root / "components" / "assistant-ui" / "tool-ui-python.tsx").read_text(
        encoding = "utf-8"
    )
    assert "export function isSandboxFileList" in adapter
    # Every entry, not just the array: the rows read name off each one.
    assert 'typeof (entry as { name?: unknown }).name === "string"' in adapter
    for source in (adapter, python_card):
        assert "isSandboxFileList(v.files)" in source


def test_a_sandbox_of_empty_directories_is_still_reclaimed(tmp_path, monkeypatch):
    """A tool that only ran mkdir, or deleted what it wrote, leaves a folder no
    chat can reach once its record is gone."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_dirs111"
    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "outputs" / "runs").mkdir(parents = True)

    assert tools.remove_session_sandbox(session) is True
    assert not workdir.exists()

    # One with a file in it still needs the opt-in.
    other = Path(tools.get_sandbox_workdir("__LOCALID_dirs222"))
    (other / "outputs").mkdir()
    (other / "outputs" / "report.csv").write_text("a,b\n", encoding = "utf-8")
    assert tools.remove_session_sandbox("__LOCALID_dirs222") is False
    assert (other / "outputs" / "report.csv").is_file()


def test_a_symlinked_builtin_cache_is_left_usable(tmp_path, monkeypatch):
    """Clearing resolves the link and removes the target, so without this the
    link dangles and the next compile cannot write through it."""
    real = tmp_path / "real_cache"
    real.mkdir()

    from utils import cache_cleanup

    (real / cache_cleanup.CACHE_MARKER).touch()
    (real / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n", encoding = "utf-8")
    link = tmp_path / "unsloth_compiled_cache"
    link.symlink_to(real, target_is_directory = True)
    monkeypatch.setattr(cache_cleanup, "_CACHE_DIRS", [link])
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)

    cache_cleanup.clear_unsloth_compiled_cache()
    assert real.is_dir(), "the link was left dangling"
    assert (real / cache_cleanup.CACHE_MARKER).is_file()
    assert not (real / "unsloth_compiled_module_gemma3.py").exists()
    # Writable through the link, which is what the compiler does next.
    os.makedirs(link, exist_ok = True)


def test_the_sandbox_listing_runs_in_a_worker():
    """It walks up to a couple of thousand entries and stats each one."""
    import inspect

    from routes import inference as inference_routes

    source = inspect.getsource(inference_routes.list_sandbox_files)
    assert "run_in_threadpool(_sandbox_listing" in source
    assert "os.stat" not in source


def test_a_marker_replaced_by_a_link_is_not_written_through(tmp_path, monkeypatch):
    """The file sits where tool code runs, so a link there would send our write
    to whatever it points at."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_link911"
    workdir = Path(tools.get_sandbox_workdir(session))
    victim = tmp_path / "notes.txt"
    victim.write_text("years of notes", encoding = "utf-8")

    marker = workdir / tools._SANDBOX_MARKER
    marker.unlink()
    marker.symlink_to(victim)
    assert tools._marker_owner(str(workdir)) is None, "followed a link to read"

    tools._mark_sandbox(str(workdir), session)
    assert victim.read_text(encoding = "utf-8") == "years of notes", "wrote through the link"
    assert not marker.is_symlink()
    assert marker.read_text(encoding = "utf-8") == session


def test_a_cached_path_swapped_for_another_chats_directory_is_dropped(tmp_path, monkeypatch):
    """cd .., mv, ln -s is all a tool needs, and containment accepts a link to
    a sibling."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    victim = Path(tools.get_sandbox_workdir("__LOCALID_victim2"))
    (victim / "private.csv").write_text("theirs", encoding = "utf-8")
    attacker = Path(tools.get_sandbox_workdir("__LOCALID_attack2"))

    shutil.rmtree(attacker)
    attacker.symlink_to(victim, target_is_directory = True)

    again = Path(tools.get_sandbox_workdir("__LOCALID_attack2"))
    assert again.resolve() != victim.resolve(), again
    assert not (again / "private.csv").exists()
    resolved = Path(tools.resolve_sandbox_workdir("__LOCALID_attack2"))
    assert not (resolved / "private.csv").exists(), resolved


def test_a_default_folder_that_was_already_there_is_not_run_in(tmp_path, monkeypatch):
    """A call with no session id lands on the fallback, which in a shared root
    can be a directory of the user's own."""
    root = tmp_path / "shared"
    root.mkdir()
    theirs = root / "_default"
    theirs.mkdir()
    (theirs / "notes.txt").write_text("theirs", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir(None))
    assert workdir != theirs, workdir
    assert not (workdir / "notes.txt").exists()
    assert (theirs / "notes.txt").read_text(encoding = "utf-8") == "theirs"
    assert not (theirs / tools._SANDBOX_MARKER).exists()

    # Ours is claimed, so the next run recognises it rather than making another.
    tools._workdirs.clear()
    assert Path(tools.get_sandbox_workdir(None)) == workdir


def test_an_id_a_path_segment_cannot_carry_rides_in_the_query():
    """ASGI decodes %2F before it matches a route, so an id with a slash in it
    would arrive as a different id and a different filename."""
    root = Path(__file__).resolve().parents[2] / "frontend" / "src"
    helper = (root / "components" / "assistant-ui" / "sandbox-files.ts").read_text(encoding = "utf-8")
    assert "export function sandboxRoutePrefix" in helper
    assert "?session=${encodeURIComponent(sessionId)}" in helper

    from routes import inference as inference_routes
    import inspect

    for route in (inference_routes.list_sandbox_files, inference_routes.serve_sandbox_file):
        source = inspect.getsource(route)
        assert "session: Optional[str] = None" in source
        assert "session or session_id" in source


def test_clearing_every_chat_reports_what_it_deleted(tmp_path, monkeypatch):
    """A thread added between the listing and the delete is gone too, and its
    sandbox has to be cleaned up with the rest."""
    import inspect

    from routes import chat_history
    from storage import studio_db

    source = inspect.getsource(studio_db.clear_chat_history)
    assert "SELECT id FROM chat_threads" in source
    assert source.index("SELECT id FROM chat_threads") < source.index("DELETE FROM chat_threads")
    assert "return removed" in source

    route = inspect.getsource(chat_history.clear_history)
    assert "cleared = clear_chat_history()" in route
    assert "cleared" in route.split("_remove_sandboxes(", 1)[1].split(")", 1)[0]


def test_a_symlinked_cache_marker_does_not_license_a_delete(tmp_path, monkeypatch):
    """exists() follows a link, so a marker pointing at any existing path would
    have the cleanup rmtree a directory of the user's own."""
    from utils import cache_cleanup

    theirs = tmp_path / "unsloth_compiled_cache"
    theirs.mkdir()
    (theirs / "notes.txt").write_text("years of notes", encoding = "utf-8")
    (theirs / cache_cleanup.CACHE_MARKER).symlink_to(tmp_path / "anything")
    (tmp_path / "anything").write_text("x", encoding = "utf-8")

    assert cache_cleanup._is_dedicated_cache(theirs) is False

    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(theirs))
    monkeypatch.chdir(tmp_path)
    cache_cleanup.clear_unsloth_compiled_cache()
    assert (theirs / "notes.txt").is_file(), "deleted a directory it does not own"


def test_a_real_cache_marker_still_counts(tmp_path):
    """The other half: a directory Studio made is still cleaned out."""
    from utils import cache_cleanup

    ours = tmp_path / "unsloth_compiled_cache"
    ours.mkdir()
    (ours / cache_cleanup.CACHE_MARKER).touch()
    assert cache_cleanup._is_dedicated_cache(ours) is True


def test_an_id_with_a_lone_surrogate_still_gets_a_directory(tmp_path, monkeypatch):
    """An API client can send one in JSON, and a POSIX name decoded with
    surrogateescape carries them too; a strict encode raises instead."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    odd = "chat-\ud800-one"
    name = tools._sandbox_name(odd)
    assert name.startswith(tools._DERIVED_PREFIX)
    # Distinct ids stay distinct: surrogatepass keeps the code point, where a
    # replacing policy would fold every bad id onto one directory.
    assert name != tools._sandbox_name("chat-\ud801-one")

    workdir = Path(tools.get_sandbox_workdir(odd))
    assert workdir.is_dir()
    assert Path(tools.resolve_sandbox_workdir(odd)) == workdir


def test_a_legacy_entry_that_is_a_symlink_is_left_alone(tmp_path, monkeypatch):
    """move() keeps the link, and the marker would then be written inside
    whatever it points at, which is outside both roots."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox"
    legacy.mkdir(parents = True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "theirs.txt").write_text("mine", encoding = "utf-8")
    (legacy / "__LOCALID_link111").symlink_to(outside, target_is_directory = True)
    (legacy / "__LOCALID_real111").mkdir()
    (legacy / "__LOCALID_real111" / "results.csv").write_text("a,b\n", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    root = Path(tools.sandbox_root())
    tools._migrate_legacy_sandbox(str(root))

    assert not (outside / tools._SANDBOX_MARKER).exists(), "wrote outside both roots"
    assert (outside / "theirs.txt").is_file()
    assert not (root / "__LOCALID_link111").exists()
    # The rest of the pass still runs.
    assert (root / "__LOCALID_real111" / "results.csv").is_file()


def test_a_first_tool_call_does_not_wait_for_the_whole_legacy_tree(tmp_path, monkeypatch):
    """Across filesystems the migration copies every session, which is minutes;
    a call that only needs its own folder must not queue behind it."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox"
    (legacy / "__LOCALID_mine111").mkdir(parents = True)
    (legacy / "__LOCALID_mine111" / "results.csv").write_text("a,b\n", encoding = "utf-8")
    (legacy / "__LOCALID_huge111").mkdir()

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    held = threading.Event()
    monkeypatch.setattr(
        tools,
        "migrate_legacy_sandbox_in_background",
        lambda: threading.Thread(target = held.wait),
    )

    class _Blocked:
        def __enter__(self):
            assert held.wait(0), "the request path took the whole-tree lock"

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(tools, "_legacy_sandbox_lock", _Blocked())
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_mine111"))
    held.set()

    assert (workdir / "results.csv").is_file(), "this chat's own files were left behind"


def test_deleting_a_chat_stops_a_generation_that_would_recreate_it(monkeypatch):
    """A request that has not reached the executor yet would dispatch its tool
    call after the delete and write files no chat can reach."""
    import routes.chat_history as chat_history

    cancelled = []
    monkeypatch.setattr(chat_history, "delete_chat_threads", lambda ids: None)
    monkeypatch.setattr(chat_history, "_cancel_active_research", lambda request, ids: None)

    from state import active_generations

    monkeypatch.setattr(active_generations, "cancel_thread", lambda tid: cancelled.append(tid) or 1)

    chat_history._cancel_active_generations(["__LOCALID_gone111", "__LOCALID_gone222"])
    assert cancelled == ["__LOCALID_gone111", "__LOCALID_gone222"]

    import inspect

    for route in (chat_history.delete_threads, chat_history.clear_history):
        source = inspect.getsource(route)
        assert "_cancel_active_generations" in source, route.__name__
        assert source.index("_cancel_active_generations") < source.index("_remove_sandboxes")


def test_a_large_file_is_streamed_rather_than_buffered():
    """A tool can write a multi-gigabyte artifact, and a Blob plus the IPC copy
    of it would be two more of it in the renderer."""
    view = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/components/assistant-ui/sandbox-files-view.tsx"
    ).read_text(encoding = "utf-8")

    assert "downloadUrlStreaming" in view
    assert "response.blob()" not in view
    # The route takes the bearer as a query parameter, since the streaming path
    # sends no headers of its own.
    assert "token=" in view


def test_a_same_size_overwrite_is_still_reported(tmp_path, monkeypatch):
    """On a coarse-timestamp volume a rewrite of the same length inside one tick
    matches on mtime and size, and the call reported no file at all."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_same111"))
    report = workdir / "report.csv"
    report.write_text("a,b\n1,2\n", encoding = "utf-8")

    before = tools._snapshot_workdir_files(str(workdir))
    report.write_text("a,b\n3,4\n", encoding = "utf-8")  # same length
    os.utime(report, ns = (before["report.csv"][0], before["report.csv"][0]))

    after = tools._snapshot_workdir_files(str(workdir))
    assert after["report.csv"][:2] == before["report.csv"][:2], "the test lost its premise"
    assert after["report.csv"] != before["report.csv"], "a same-size rewrite looked unchanged"

    sentinel = tools._created_file_sentinels(str(workdir), before)
    assert "report.csv" in (sentinel or ""), sentinel


def test_a_file_too_big_to_read_is_still_snapshotted(tmp_path, monkeypatch):
    """The digest is bounded: reading every artifact twice per call is not the
    price for a case that needs a coarse clock and an exact length match."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_big1111"))
    big = workdir / "model.bin"
    big.write_bytes(b"0" * (tools._MAX_HASHED_SNAPSHOT_BYTES + 1))

    entry = tools._snapshot_workdir_files(str(workdir))["model.bin"]
    assert entry[1] == tools._MAX_HASHED_SNAPSHOT_BYTES + 1
    assert entry[2] is None, "hashed a file past the cap"


def test_deleting_a_big_sandbox_does_not_hold_up_other_chats(tmp_path, monkeypatch):
    """Every tool start takes this lock, so an rmtree in here stops calls in
    every unrelated chat for as long as it runs."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_big2222"
    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "data.csv").write_text("a,b\n", encoding = "utf-8")

    started = threading.Event()
    release = threading.Event()
    real_rmtree = tools.shutil.rmtree

    def slow_rmtree(path, **kwargs):
        started.set()
        release.wait(10)
        return real_rmtree(path, **kwargs)

    monkeypatch.setattr(tools.shutil, "rmtree", slow_rmtree)
    outcome = []
    deleter = threading.Thread(
        target = lambda: outcome.append(tools.remove_session_sandbox(session, delete_files = True)),
    )
    try:
        deleter.start()
        assert started.wait(5), "the delete never started"
        # Observed while the tree is still going: this is the lock every tool
        # start takes, so holding it here stops calls in every other chat.
        assert tools._active_sessions_lock.acquire(timeout = 2), "held the tool lock"
        tools._active_sessions_lock.release()
        assert not workdir.exists(), "the name is still there"
    finally:
        release.set()
        deleter.join(15)
        monkeypatch.setattr(tools.shutil, "rmtree", real_rmtree)
    assert outcome == [True], outcome
    for _ in range(100):
        leftovers = [n for n in os.listdir(tools.sandbox_root()) if tools._DETACHED_SUFFIX in n]
        if not leftovers:
            break
        time.sleep(0.05)
    assert leftovers == [], leftovers


def test_an_interrupted_move_is_not_read_as_a_collision(tmp_path, monkeypatch):
    """Across filesystems the copy fills the destination as it goes, so a run
    killed part way leaves a partial directory with the source still there."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox" / "__LOCALID_part111"
    legacy.mkdir(parents = True)
    (legacy / "results.csv").write_text("a,b\n", encoding = "utf-8")
    (legacy / "second.csv").write_text("c,d\n", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    root = Path(tools.sandbox_root())
    root.mkdir(parents = True, exist_ok = True)

    real_move = tools.shutil.move

    def half_copied(source, target):
        os.makedirs(target, exist_ok = True)
        shutil.copy2(os.path.join(source, "results.csv"), os.path.join(target, "results.csv"))
        raise OSError(5, "interrupted")

    monkeypatch.setattr(tools.shutil, "move", half_copied)
    tools._migrate_legacy_sandbox(str(root))
    monkeypatch.setattr(tools.shutil, "move", real_move)

    # Nothing at the session's name, so the next launch does not read the half
    # copy as a session the new root already has.
    assert not (root / "__LOCALID_part111").exists(), "left a partial copy at the real name"
    assert [n for n in os.listdir(root) if tools._STAGING_SUFFIX in n] == []
    assert (legacy / "second.csv").is_file(), "lost the source"

    tools._legacy_sandbox_migrated = False
    tools._migrate_legacy_sandbox(str(root))
    assert (root / "__LOCALID_part111" / "second.csv").is_file(), "the retry never happened"


def test_a_delete_without_the_switch_says_what_it_kept(tmp_path, monkeypatch):
    """Surfaces other than the sidebar never offer the choice, and after the
    delete the folder is unreachable, so the route reports it."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    kept = "__LOCALID_keeps11"
    empty = "__LOCALID_empty11"
    workdir = Path(tools.get_sandbox_workdir(kept))
    (workdir / "results.csv").write_text("a,b\n", encoding = "utf-8")
    Path(tools.get_sandbox_workdir(empty))

    assert tools.session_sandbox_has_files(kept) is True
    assert tools.session_sandbox_has_files(empty) is False
    assert tools.session_sandbox_has_files("__LOCALID_never11") is False

    import asyncio

    import routes.chat_history as chat_history

    removed, still_there = (
        asyncio.get_event_loop_policy()
        .new_event_loop()
        .run_until_complete(chat_history._remove_sandboxes([kept, empty], False))
    )
    assert removed == 1, removed  # the empty one
    assert still_there == [kept], still_there
    assert (workdir / "results.csv").is_file()


def test_every_delete_surface_can_still_reach_the_files():
    """The offer is made where every surface goes through, not only the one
    dialog that has the switch."""
    src = Path(__file__).resolve().parents[2] / "frontend/src"
    hook = (src / "features/chat/hooks/use-chat-sidebar-items.ts").read_text(
        encoding = "utf-8",
    )
    offer = (src / "features/chat/utils/offer-kept-sandbox-files.ts").read_text(
        encoding = "utf-8",
    )

    body = hook[hook.index("export async function deleteChatItem") :]
    assert "offerToDeleteKeptSandboxes(kept)" in body
    assert "toast(" in offer, "nothing tells the user the files are still there"
    assert "deleteFiles: true" in offer


def test_a_fallback_name_already_in_a_shared_root_is_not_taken(tmp_path, monkeypatch):
    """Both names can be the user's, and claiming the second one puts the chat
    inside their files, which a delete with the switch would then remove."""
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_both111"
    name = tools._sandbox_name(session)
    theirs = root / name
    theirs.mkdir()
    (theirs / "plain.txt").write_text("mine", encoding = "utf-8")
    fallback = root / f"{name}-{tools._name_suffix(session)}"
    fallback.mkdir()
    (fallback / "also-mine.txt").write_text("mine", encoding = "utf-8")

    workdir = Path(tools.get_sandbox_workdir(session))
    assert workdir not in (theirs, fallback), workdir
    assert not (theirs / tools._SANDBOX_MARKER).exists()
    assert not (fallback / tools._SANDBOX_MARKER).exists()
    assert tools._marker_owner(str(workdir)) == name

    # And a delete takes only what we made.
    tools.remove_session_sandbox(session, delete_files = True)
    assert (theirs / "plain.txt").is_file()
    assert (fallback / "also-mine.txt").is_file()


def test_a_symlinked_sandbox_marker_does_not_make_a_directory_ours(tmp_path, monkeypatch):
    """isfile() follows the link, so a marker pointing at any existing file
    would let a delete remove an unrelated directory in a shared root."""
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_fake111"
    theirs = root / tools._sandbox_name(session)
    theirs.mkdir()
    (theirs / "notes.txt").write_text("years of notes", encoding = "utf-8")
    (root / "anything").write_text("x", encoding = "utf-8")
    (theirs / tools._SANDBOX_MARKER).symlink_to(root / "anything")

    assert tools._sandbox_is_ours(str(theirs)) is False
    assert tools.remove_session_sandbox(session, delete_files = True) is False
    assert (theirs / "notes.txt").is_file(), "deleted a directory it does not own"


def test_a_chat_that_owns_nothing_never_reads_from_the_shared_root(tmp_path, monkeypatch):
    """The reserved-looking path was inside the user's own root, so a folder
    they happen to keep at that name would have been listed and served."""
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_none111"
    theirs = root / tools._sandbox_name(session)
    theirs.mkdir()
    (theirs / "private.csv").write_text("theirs", encoding = "utf-8")
    # What the old sentinel pointed at, filled by the user.
    planted = root / "_unowned" / tools._sandbox_name(session)
    planted.mkdir(parents = True)
    (planted / "also-theirs.csv").write_text("theirs", encoding = "utf-8")

    # The fallback name is theirs too, so the resolver has nothing of ours to
    # answer with and must not point at anything in here.
    fallback = root / f"{tools._sandbox_name(session)}-{tools._name_suffix(session)}"
    fallback.mkdir()
    (fallback / "third.csv").write_text("theirs", encoding = "utf-8")

    resolved = Path(tools.resolve_sandbox_workdir(session))
    assert not str(resolved).startswith(str(root)), resolved
    assert not (resolved / "also-theirs.csv").exists()
    assert not (resolved / "private.csv").exists()
    assert not (resolved / "third.csv").exists()


def test_a_request_path_move_lands_where_the_resolver_says(tmp_path, monkeypatch):
    """Stopping at the plain-name collision left the chat with an empty sandbox
    and its files at the old root, with the whole-tree pass then agreeing."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox" / "__LOCALID_taken11"
    legacy.mkdir(parents = True)
    (legacy / "results.csv").write_text("a,b\n", encoding = "utf-8")

    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    theirs = root / "__LOCALID_taken11"
    theirs.mkdir()
    (theirs / "plain.txt").write_text("mine", encoding = "utf-8")

    tools._migrate_one_legacy_session(str(root), "__LOCALID_taken11")

    assert (theirs / "plain.txt").is_file(), "moved into the user's own folder"
    landed = Path(tools.get_sandbox_workdir("__LOCALID_taken11"))
    assert (landed / "results.csv").is_file(), f"the chat's files were stranded: {landed}"


def test_deleting_a_project_takes_its_chats_sandboxes(tmp_path, monkeypatch):
    """A chat can write files before it joins a project, and deleting the
    project removes the only records of those chats."""
    import inspect

    from routes import chat_history

    source = inspect.getsource(chat_history.delete_project)
    assert "_remove_sandboxes(member_ids" in source
    assert "_cancel_active_generations(member_ids)" in source
    assert source.index("delete_chat_project(") < source.index(
        "_cancel_active_generations(member_ids)"
    )


def test_a_native_download_gets_an_absolute_url():
    """The native command parses the URL and rejects a relative one, so a bare
    /api path failed before the request was made."""
    view = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/components/assistant-ui/sandbox-files-view.tsx"
    ).read_text(encoding = "utf-8")

    assert "apiUrl(" in view
    body = view[view.index("const save = useCallback") :]
    assert body.index("apiUrl(") < body.index("downloadUrlStreaming(")


def _forget_sandbox_state(tools):
    """Everything a fresh process would not remember."""
    tools._workdirs.clear()
    getattr(tools, "_claimed_here", set()).clear()


def _shared_root(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))
    return root


def test_a_fallback_with_a_random_name_is_found_again(tmp_path, monkeypatch):
    """Nothing can recompute that name, so without the marker the chat gets a
    new folder every launch and a delete never reaches the old ones."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "__LOCALID_rand111"
    name = tools._sandbox_name(session)
    for taken in (root / name, root / f"{name}-{tools._name_suffix(session)}"):
        taken.mkdir()

    first = Path(tools.get_sandbox_workdir(session))
    (first / "report.csv").write_text("a,b\n", encoding = "utf-8")
    assert tools._marker_owner(str(first)) == name

    # A later launch: nothing cached, and the name is not derivable.
    _forget_sandbox_state(tools)
    assert Path(tools.get_sandbox_workdir(session)) == first
    assert Path(tools.resolve_sandbox_workdir(session)) == first

    assert tools.remove_session_sandbox(session, delete_files = True) is True
    assert not first.exists(), "the delete could not reach the fallback"


def test_a_marker_a_tool_deleted_is_written_again(tmp_path, monkeypatch):
    """Tool code runs in this directory. Reading the missing marker as somebody
    else's strands the files already written and restarts the chat elsewhere."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "__LOCALID_clob111"
    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "plot.png").write_bytes(b"x")
    (workdir / tools._SANDBOX_MARKER).unlink()

    again = Path(tools.get_sandbox_workdir(session))
    assert again == workdir, "the chat walked away from its own files"
    assert tools._marker_owner(str(workdir)) == tools._sandbox_name(session)
    assert (again / "plot.png").is_file()

    # A directory this run never claimed is still left alone.
    theirs = root / "not-ours"
    theirs.mkdir()
    tools._workdirs[("__LOCALID_other11", None)] = str(theirs)
    assert Path(tools.get_sandbox_workdir("__LOCALID_other11")) != theirs
    assert not (theirs / tools._SANDBOX_MARKER).exists()


def test_an_interrupted_delete_is_finished_on_the_next_launch(tmp_path, monkeypatch):
    """The rename is what puts the tree out of reach, so a kill before the
    rmtree leaves a full copy of the files nothing resolves to."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    tools._workdirs.clear()
    tombstone = root / f"__LOCALID_gone111{tools._DETACHED_SUFFIX}0123abcd"
    tombstone.mkdir()
    (tombstone / "secret.csv").write_text("rows", encoding = "utf-8")
    (tombstone / tools._SANDBOX_MARKER).write_text(
        "__LOCALID_gone111",
        encoding = "utf-8",
    )
    # The user's own, named similarly and never marked.
    theirs = root / "report.deleting-old"
    theirs.mkdir()
    (theirs / "keep.txt").write_text("mine", encoding = "utf-8")

    tools.sweep_detached_sandboxes(str(root))

    assert not tombstone.exists(), "an interrupted delete left the files behind"
    assert (theirs / "keep.txt").is_file(), "swept a folder it does not own"


def test_a_legacy_chat_named_like_a_derived_id_is_migrated(tmp_path, monkeypatch):
    """Its folder is under the literal id, but the name is now hashed, so
    looking only at the hash leaves those files at the old root."""
    fake_home = tmp_path / "userprofile"
    session = tools_derived_id = "_id-0123456789abcdef"
    legacy = fake_home / "studio_sandbox" / session
    legacy.mkdir(parents = True)
    (legacy / "old.csv").write_text("a,b\n", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    assert tools_derived_id.startswith(tools._DERIVED_PREFIX)
    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False

    landed = Path(tools.get_sandbox_workdir(session))
    assert (landed / "old.csv").is_file(), f"files stranded at the legacy root: {landed}"


def test_clearing_every_chat_reports_the_files_it_kept():
    """Without the switch the folders survive the clear, and afterwards there
    is no card left to reach them from."""
    src = Path(__file__).resolve().parents[2] / "frontend/src"

    api = (src / "features/chat/api/chat-api.ts").read_text(encoding = "utf-8")
    clear = api[api.index("export async function clearBackendChats") :]
    clear = clear[: clear.index("\nexport ")]
    assert "Promise<string[]>" in clear
    assert "sandboxes_kept" in clear

    storage = (src / "features/chat/utils/chat-history-storage.ts").read_text(
        encoding = "utf-8",
    )
    assert "sandboxesKept" in storage
    assert "result.sandboxesKept = await clearBackendChats" in storage

    tab = (src / "features/settings/tabs/data-tab.tsx").read_text(encoding = "utf-8")
    assert "offerToDeleteKeptSandboxes(result.sandboxesKept)" in tab

    offer = (src / "features/chat/utils/offer-kept-sandbox-files.ts").read_text(
        encoding = "utf-8",
    )
    assert "deleteFiles: true" in offer
    # And the per-chat surfaces go through the same offer.
    hook = (src / "features/chat/hooks/use-chat-sidebar-items.ts").read_text(
        encoding = "utf-8",
    )
    assert "offerToDeleteKeptSandboxes(kept)" in hook


def test_a_lone_surrogate_id_can_still_step_aside(tmp_path, monkeypatch):
    """The collision path encodes the id a second time, and a strict encode
    there raises before the chat can be given a name of its own."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "chat-\udce2-1"  # what an API client can send, and what os.listdir returns
    theirs = root / tools._sandbox_name(session)
    theirs.mkdir()
    (theirs / "theirs.txt").write_text("mine", encoding = "utf-8")

    workdir = Path(tools.get_sandbox_workdir(session))
    assert workdir != theirs
    assert tools._marker_owner(str(workdir)) == tools._sandbox_name(session)


def test_a_legacy_move_lands_when_both_names_are_taken(tmp_path, monkeypatch):
    """Both derived names can be the user's, and returning there left the files
    at the legacy root with nothing that would ever move them."""
    fake_home = tmp_path / "userprofile"
    session = "__LOCALID_both222"
    legacy = fake_home / "studio_sandbox" / session
    legacy.mkdir(parents = True)
    (legacy / "results.csv").write_text("a,b\n", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False
    taken = [root / session, root / f"{session}-{tools._name_suffix(session)}"]
    for directory in taken:
        directory.mkdir()
        (directory / "theirs.txt").write_text("mine", encoding = "utf-8")

    tools._migrate_one_legacy_session(str(root), session)

    landed = Path(tools.get_sandbox_workdir(session))
    assert (landed / "results.csv").is_file(), f"files stranded at the legacy root: {landed}"
    for directory in taken:
        assert (directory / "theirs.txt").is_file(), "moved into the user's own folder"


def test_a_read_finds_the_marked_fallback_after_a_restart(tmp_path, monkeypatch):
    """Only creation and deletion scanned for the marker, so every file card in
    the transcript 404s until some later tool call refills the cache."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "__LOCALID_read111"
    name = tools._sandbox_name(session)
    for taken in (root / name, root / f"{name}-{tools._name_suffix(session)}"):
        taken.mkdir()

    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "report.csv").write_text("a,b\n", encoding = "utf-8")

    _forget_sandbox_state(tools)  # a restart: nothing cached
    served = Path(tools.resolve_sandbox_workdir(session))
    assert served == workdir, f"the read could not find the fallback: {served}"
    assert (served / "report.csv").is_file()


def test_a_chat_started_during_the_clear_is_cancelled_too():
    """Its id is in the transaction's result and its sandbox is removed, but a
    generation still running would dispatch a tool and rebuild it."""
    import inspect

    from routes import chat_history

    source = inspect.getsource(chat_history.clear_history)
    assert "late" in source
    assert source.index("cleared = clear_chat_history()") < source.index(
        "_cancel_active_generations(late)"
    )
    assert source.index("_cancel_active_generations(late)") < source.index("_remove_sandboxes(")


def test_one_call_never_reports_another_calls_scratch_script(tmp_path, monkeypatch):
    """Chats in one project share a workdir, and each call snapshots the whole
    directory, so the other call's studio_exec_*.py was offered as a download
    that 404s the moment that call cleans it up."""
    from core.inference import tools

    workdir = tmp_path / "shared-workdir"
    workdir.mkdir()
    before = tools._snapshot_workdir_files(str(workdir))

    (workdir / "studio_exec_abc123.py").write_text("print(1)", encoding = "utf-8")
    (workdir / "report.csv").write_text("a,b\n", encoding = "utf-8")
    with tools._scratch_lock:
        tools._active_scratch.add("studio_exec_abc123.py")
    try:
        sentinels = tools._created_file_sentinels(str(workdir), before)
    finally:
        with tools._scratch_lock:
            tools._active_scratch.discard("studio_exec_abc123.py")

    assert "report.csv" in sentinels
    assert "studio_exec_abc123.py" not in sentinels, sentinels


def test_a_program_cannot_print_its_own_file_envelope(tmp_path, monkeypatch):
    """A call that created nothing appends no envelope, so a printed one is the
    last marker in the result and is read as ours: the line disappears from the
    model's view and the UI offers a download for a file nobody wrote."""
    from core.inference import tools
    from core.inference.tool_loop_controller import strip_result_for_model

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    _forget_sandbox_state(tools)

    forged = '__FILES__:[{"name": "payroll.csv", "size": 12}]'
    # After a line of its own: both readers anchor the marker to a line start.
    result = tools._python_exec(
        f"print('working')\nprint({forged!r})",
        session_id = "__LOCALID_forge11",
    )

    assert "payroll.csv" in result, result  # the text itself is still shown
    assert "\n__FILES__:" not in result, result
    assert (
        strip_result_for_model(result).count("payroll.csv") == 1
    ), "the printed line was eaten as an envelope"


def test_a_delete_that_waited_for_a_tool_call_says_it_kept_the_files(tmp_path, monkeypatch):
    """The sandbox can be empty at the moment of the check and hold a file a
    second later, and the deferred removal keeps it with nobody left to ask."""
    import asyncio

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools
    from routes import chat_history

    _forget_sandbox_state(tools)
    session = "__LOCALID_busy111"
    workdir = Path(tools.get_sandbox_workdir(session))
    assert workdir.is_dir()

    with tools._session_in_flight(session):  # a tool call running in this chat
        removed, kept = asyncio.new_event_loop().run_until_complete(
            chat_history._remove_sandboxes([session], False)
        )
        assert tools.sandbox_removal_deferred(session) is True
        assert removed == 0, removed
        assert kept == [session], "an unreachable folder with no offer to delete it"


def test_a_completed_move_is_never_the_thing_that_gets_deleted(tmp_path, monkeypatch):
    """shutil.move already took the legacy copy, so the staging tree is the only
    one there is and deleting it loses the user's files for good."""
    fake_home = tmp_path / "userprofile"
    session = "__LOCALID_lost111"
    legacy = fake_home / "studio_sandbox" / session
    legacy.mkdir(parents = True)
    (legacy / "thesis.txt").write_text("years of work", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    root = tmp_path / "home" / "studio_sandbox"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    root.mkdir(parents = True)

    real_rename = os.rename

    def failing_rename(src, dst):
        if str(src).find(tools._STAGING_SUFFIX) != -1:
            raise OSError("rename refused")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", failing_rename)
    with pytest.raises(OSError):
        tools._staged_move(str(legacy), str(root / session), session)
    monkeypatch.setattr(os, "rename", real_rename)

    survivors = [p for p in [legacy, *root.iterdir()] if (p / "thesis.txt").is_file()]
    assert survivors, "the only copy of the user's files was deleted"


def test_a_symlinked_directory_counts_as_a_file_of_the_users(tmp_path, monkeypatch):
    """os.walk lists it in dirs, and a check that reads only files called the
    sandbox empty and removed the link the tool made."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "__LOCALID_link111"
    workdir = Path(tools.get_sandbox_workdir(session))
    elsewhere = tmp_path / "data"
    elsewhere.mkdir()
    (workdir / "dataset").symlink_to(elsewhere, target_is_directory = True)

    assert tools._holds_no_user_files(str(workdir)) is False
    assert tools.remove_session_sandbox(session, delete_files = False) is False
    assert (workdir / "dataset").is_symlink(), "removed something the tool made"
    assert tools.session_sandbox_has_files(session) is True


def test_a_project_delete_uses_the_membership_it_really_deleted():
    """A chat moved in after the listing is deleted by the transaction, and its
    generation would keep running and rebuild a sandbox nothing can reach."""
    import inspect

    from routes import chat_history
    from storage import studio_db

    storage = inspect.getsource(studio_db.delete_chat_project)
    assert 'project["memberIds"] = sorted(thread_ids)' in storage

    route = inspect.getsource(chat_history.delete_project)
    # Only the transaction's membership: a chat moved out just before it
    # survives the delete, and cancelling or deleting from an earlier listing
    # would stop it and remove the files it wrote.
    assert 'member_ids = list(project.get("memberIds") or [])' in route
    assert "list_chat_threads(project_id" not in route
    assert route.index("_cancel_active_generations(member_ids)") < route.index("_remove_sandboxes(")
    # And what survived is reported, or the folders are reachable from nothing.
    assert "sandboxes_kept = await _remove_sandboxes(member_ids" in route
    assert "ChatProjectDeleted(**project, sandboxes_kept = sandboxes_kept)" in route


def test_closing_an_incognito_chat_cleans_up_its_sandbox():
    """Its id is what the tool call sent as the sandbox session, so a folder
    exists on disk even though no history row does."""
    src = Path(__file__).resolve().parents[2] / "frontend/src"

    storage = (src / "features/chat/utils/chat-history-storage.ts").read_text(
        encoding = "utf-8",
    )
    body = storage[storage.index("export async function deleteStoredChatThreads") :]
    body = body[: body.index("\nexport ")]
    assert "deleteChatThreads(idsToDelete" in body, "incognito ids never reach the backend"
    assert "isThreadIncognito" in body  # the Dexie work still skips them

    projects = (src / "features/chat/hooks/use-chat-projects.ts").read_text(
        encoding = "utf-8",
    )
    assert "offerToDeleteKeptSandboxes(kept)" in projects


def test_a_snapshot_stops_hashing_once_it_has_read_enough(tmp_path, monkeypatch):
    """The file cap alone allowed thousands of files under the per-file bound to
    be read in full, twice per call, on a directory of experiment artifacts."""
    from core.inference import tools

    workdir = tmp_path / "artifacts"
    workdir.mkdir()
    chunk = b"x" * (1024 * 1024)
    for i in range(8):
        (workdir / f"run{i}.bin").write_bytes(chunk)

    monkeypatch.setattr(tools, "_MAX_SNAPSHOT_HASH_BYTES", 3 * 1024 * 1024)
    read = []
    real_content_key = tools._content_key

    def counting_content_key(path, size):
        read.append(size)
        return real_content_key(path, size)

    monkeypatch.setattr(tools, "_content_key", counting_content_key)
    snapshot = tools._snapshot_workdir_files(str(workdir))

    assert len(snapshot) == 8, snapshot  # every file is still reported
    assert sum(read) <= 3 * 1024 * 1024 + 1024 * 1024, sum(read)
    assert any(key[2] is None for key in snapshot.values()), "nothing fell back"


def test_a_call_that_shared_its_workdir_claims_nothing(tmp_path):
    """Chats in one project share a workdir. Each call diffs the whole tree, so
    the other call's output was advertised on this card and its download served
    that content. No timestamps: a coarse or remote clock is exactly what this
    cannot depend on."""
    from core.inference import tools

    workdir = tmp_path / "project-workspace"
    workdir.mkdir()

    theirs = tools._call_started(str(workdir))  # the other chat's call
    try:
        before = tools._snapshot_workdir_files(str(workdir))
        ours = tools._call_started(str(workdir))  # ours starts while theirs runs
        try:
            (workdir / "theirs.csv").write_text("a,b\n", encoding = "utf-8")
            sentinels = tools._created_file_sentinels(str(workdir), before, None, ours)
        finally:
            tools._call_finished(ours)
        assert ours["shared"] is True
        assert theirs["shared"] is True, "the call already running was not told"
    finally:
        tools._call_finished(theirs)

    assert sentinels == "", sentinels

    # Alone in the workdir, the same write is reported as before.
    alone = tools._call_started(str(workdir))
    try:
        before = tools._snapshot_workdir_files(str(workdir))
        (workdir / "ours.csv").write_text("a,b\n", encoding = "utf-8")
        sentinels = tools._created_file_sentinels(str(workdir), before, None, alone)
    finally:
        tools._call_finished(alone)
    assert "ours.csv" in sentinels, sentinels


def test_a_read_serves_the_legacy_files_while_the_move_is_still_running(tmp_path, monkeypatch):
    """The move runs in the background and across filesystems takes minutes, and
    a pass that fails leaves the files there for the rest of the process."""
    fake_home = tmp_path / "userprofile"
    session = "__LOCALID_slow111"
    legacy = fake_home / "studio_sandbox" / session
    legacy.mkdir(parents = True)
    (legacy / "plot.png").write_bytes(b"png")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False

    served = Path(tools.resolve_sandbox_workdir(session))
    assert (served / "plot.png").is_file(), f"the card 404s until a later tool call: {served}"
    # And nothing was created at the new root by the read.
    assert not (Path(tools.sandbox_root()) / session).exists()


def test_a_case_variant_id_cannot_delete_a_markerless_sandbox(tmp_path, monkeypatch):
    """On Windows and a default macOS volume `Foo` and `foo` are one directory,
    and with the marker gone the default root said yes to either."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    workdir = Path(tools.get_sandbox_workdir("Foo_chat1"))
    (workdir / "notes.txt").write_text("theirs", encoding = "utf-8")
    (workdir / tools._SANDBOX_MARKER).unlink()  # a tool wrote over it
    _forget_sandbox_state(tools)

    # This host is case-sensitive, so the volume that folds the two names is
    # modelled where the folding happens: both ids resolve to the one directory.
    real_session_dir = tools._session_dir
    monkeypatch.setattr(
        tools,
        "_session_dir",
        lambda root, session_id: (
            str(workdir)
            if session_id.casefold() == "foo_chat1"
            else real_session_dir(root, session_id)
        ),
    )

    assert tools.remove_session_sandbox("foo_chat1", delete_files = True) is False
    assert (workdir / "notes.txt").is_file(), "deleted another chat's files"
    # Its own id still reaches it.
    assert tools.remove_session_sandbox("Foo_chat1", delete_files = True) is True


def test_a_delete_moves_only_its_own_session_up(tmp_path, monkeypatch):
    """The whole-tree pass is a cross-filesystem copy of every chat, and the
    delete used to sit behind it before its response could go out."""
    fake_home = tmp_path / "userprofile"
    legacy_root = fake_home / "studio_sandbox"
    session = "__LOCALID_quick11"
    (legacy_root / session).mkdir(parents = True)
    (legacy_root / session / "mine.txt").write_text("x", encoding = "utf-8")
    for other in range(3):
        (legacy_root / f"__LOCALID_other{other}").mkdir()

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False
    monkeypatch.setattr(tools, "_start_legacy_migration", lambda: None)
    whole_tree = []
    monkeypatch.setattr(tools, "_migrate_legacy_sandbox", lambda root: whole_tree.append(root))

    tools.remove_session_sandbox(session, delete_files = True)

    assert whole_tree == [], "the delete waited for every other chat to move"
    assert not (legacy_root / session).exists(), "its own files were left behind"
    for other in range(3):
        assert (legacy_root / f"__LOCALID_other{other}").is_dir()


def test_a_deferred_removal_runs_outside_the_global_lock(tmp_path, monkeypatch):
    """It walks the tree to decide whether to keep the files, and every tool
    call in every other chat waits on that lock to start."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "__LOCALID_defer11"
    tools.get_sandbox_workdir(session)

    held = []

    def slow_remove(session_id, delete_files):
        held.append(tools._active_sessions_lock.acquire(blocking = False))
        if held[-1]:
            tools._active_sessions_lock.release()
        return True

    monkeypatch.setattr(tools, "_remove_session_sandbox_locked", slow_remove)
    with tools._session_in_flight(session):
        tools.remove_session_sandbox(session, delete_files = True)

    assert held == [True], "the removal ran while holding the global lock"


def test_only_the_sandbox_tools_have_their_file_line_read(tmp_path):
    """An MCP tool or a fetched page can legitimately end in a well-formed
    __FILES__ line, and stripping it takes that content from the model."""
    from core.inference.tool_loop_controller import strip_result_for_model

    printed = 'here is the manifest\n__FILES__:[{"name": "report.csv", "size": 1}]'

    assert "__FILES__" not in strip_result_for_model(printed, "python")
    assert "__FILES__" not in strip_result_for_model(printed, "terminal")
    assert strip_result_for_model(printed, "mcp__files__list") == printed
    assert strip_result_for_model(printed, "web_search") == printed


def test_a_migration_that_could_not_move_in_is_adopted(tmp_path, monkeypatch):
    """The rename failed after the tree had already moved, so the marked
    staging directory is the only copy of the user's files there is."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "__LOCALID_stage11"
    root = Path(tools.sandbox_root())
    root.mkdir(parents = True, exist_ok = True)
    stranded = root / f"{session}{tools._STAGING_SUFFIX}0123abcd"
    stranded.mkdir()
    (stranded / "thesis.txt").write_text("years of work", encoding = "utf-8")
    tools._mark_sandbox(str(stranded), session)

    served = Path(tools.resolve_sandbox_workdir(session))
    assert (served / "thesis.txt").is_file(), f"the only copy was unreachable: {served}"

    workdir = Path(tools.get_sandbox_workdir(session))
    assert (workdir / "thesis.txt").is_file(), "the next tool call started in an empty folder"


def test_bulk_deletes_share_one_sweeper(tmp_path, monkeypatch):
    """A clear-all can hand over every chat at once, and a thread with a
    recursive walk per chat is what exhausts the process."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    started = []
    real_thread = threading.Thread

    class CountingThread(real_thread):
        def start(self):
            started.append(self.name)
            super().start()

    monkeypatch.setattr(threading, "Thread", CountingThread)
    monkeypatch.setattr(tools.threading, "Thread", CountingThread)
    tools._delete_worker = None

    for i in range(12):
        session = f"__LOCALID_bulk{i:03d}"
        workdir = Path(tools.get_sandbox_workdir(session))
        (workdir / "out.csv").write_text("a,b\n", encoding = "utf-8")
        assert tools.remove_session_sandbox(session, delete_files = True) is True

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not any(name.startswith(f"__LOCALID_bulk") for name in os.listdir(tools.sandbox_root())):
            break
        time.sleep(0.05)

    assert started.count("sandbox-delete") == 1, started
    for i in range(12):
        assert not (Path(tools.sandbox_root()) / f"__LOCALID_bulk{i:03d}").exists()


def test_one_chats_legacy_copy_does_not_hold_up_another(tmp_path, monkeypatch):
    """A single lock around every move meant a first tool call in one chat sat
    behind a multi-gigabyte copy belonging to a different one."""
    fake_home = tmp_path / "userprofile"
    legacy_root = fake_home / "studio_sandbox"
    for name in ("__LOCALID_huge111", "__LOCALID_tiny111"):
        (legacy_root / name).mkdir(parents = True)
        (legacy_root / name / "data.bin").write_bytes(b"x")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False
    getattr(tools, "_legacy_session_locks", {}).clear()

    started = threading.Event()
    release = threading.Event()
    real_move = tools._staged_move

    def slow_move(source, target, name):
        if "huge" in name:
            started.set()
            assert release.wait(timeout = 5)
        return real_move(source, target, name)

    monkeypatch.setattr(tools, "_staged_move", slow_move)

    big = threading.Thread(
        target = tools._migrate_one_legacy_session,
        args = (tools.sandbox_root(), "__LOCALID_huge111"),
    )
    big.start()
    try:
        assert started.wait(timeout = 5)
        small = threading.Thread(
            target = tools._migrate_one_legacy_session,
            args = (tools.sandbox_root(), "__LOCALID_tiny111"),
        )
        small.start()
        small.join(timeout = 5)
        assert not small.is_alive(), "the small chat waited on the big chat's copy"
        assert (Path(tools.sandbox_root()) / "__LOCALID_tiny111" / "data.bin").is_file()
    finally:
        release.set()
        big.join(timeout = 5)


def test_an_absolute_session_id_cannot_reach_outside_the_sentinel(tmp_path, monkeypatch):
    """The id comes straight from the query, and os.path.join drops the root it
    is given when the second half is absolute."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._NOTHING_ROOT = None

    for hostile in ("/etc", "/", os.path.join("..", "..", "etc")):
        resolved = Path(tools._nothing_to_serve(hostile))
        assert not (resolved / "passwd").exists(), resolved
        assert str(resolved).startswith(tempfile.gettempdir()), resolved

    # And through the resolver, which is what the download route asks.
    session = "/etc"
    theirs = root / tools._sandbox_name(session)
    theirs.mkdir()
    served = Path(tools.resolve_sandbox_workdir(session))
    assert not str(served).startswith("/etc"), served
    assert str(served) != "/", served


def test_a_legacy_copy_of_a_folder_we_already_moved_is_left_alone(tmp_path, monkeypatch):
    """The destination already carries this chat's marker from an earlier move,
    so a second one would take the old copy somewhere nothing resolves to."""
    fake_home = tmp_path / "userprofile"
    session = "__LOCALID_twice11"
    legacy = fake_home / "studio_sandbox" / session
    legacy.mkdir(parents = True)
    (legacy / "old.csv").write_text("old", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False
    getattr(tools, "_legacy_session_locks", {}).clear()
    # What an earlier move left: the destination, marked, already in place.
    moved = root / session
    moved.mkdir()
    (moved / "new.csv").write_text("new", encoding = "utf-8")
    tools._mark_sandbox(str(moved), session)

    tools._migrate_one_legacy_session(str(root), session)

    assert (legacy / "old.csv").is_file(), "the legacy copy went somewhere unreachable"
    assert Path(tools.resolve_sandbox_workdir(session)) == moved
    assert len(list(root.iterdir())) == 1, list(root.iterdir())


def test_an_interrupted_delete_is_swept_even_without_its_marker(tmp_path, monkeypatch):
    """A tool can remove the marker before the delete, and nothing but this code
    names a directory that way in our own root."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    root = Path(tools.sandbox_root())
    root.mkdir(parents = True, exist_ok = True)
    tombstone = root / f"__LOCALID_gone222{tools._DETACHED_SUFFIX}0123abcd"
    tombstone.mkdir()
    (tombstone / "big.bin").write_bytes(b"x" * 16)

    tools.sweep_detached_sandboxes(str(root))
    assert not tombstone.exists(), "an unreachable tree was left on disk for good"


def test_the_startup_pass_lands_a_chat_whose_names_are_taken(tmp_path, monkeypatch):
    """It skipped the collision and still reported itself complete, which turns
    off both legacy reads and the per-session retry: those files are gone."""
    fake_home = tmp_path / "userprofile"
    session = "__LOCALID_startup1"
    legacy = fake_home / "studio_sandbox" / session
    legacy.mkdir(parents = True)
    (legacy / "notes.csv").write_text("a,b\n", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False
    for taken in (root / session, root / f"{session}-{tools._name_suffix(session)}"):
        taken.mkdir()
        (taken / "theirs.txt").write_text("mine", encoding = "utf-8")

    tools._migrate_legacy_sandbox(str(root))

    assert tools._legacy_sandbox_migrated is True
    landed = Path(tools.resolve_sandbox_workdir(session))
    assert (landed / "notes.csv").is_file(), f"the chat's files were stranded: {landed}"


def test_a_fallback_is_found_in_a_root_full_of_other_folders(tmp_path, monkeypatch):
    """The scan is bounded and sorted, so a big enough root hid the fallback and
    the next launch made another one beside it."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "zz_LOCALID_late11"
    name = tools._sandbox_name(session)
    for taken in (root / name, root / f"{name}-{tools._name_suffix(session)}"):
        taken.mkdir()

    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "report.csv").write_text("a,b\n", encoding = "utf-8")

    # Everything the scan would reach before this chat's own folder.
    monkeypatch.setattr(tools, "_MAX_SNAPSHOT_DIRS", 4)
    for i in range(40):
        (root / f"aaa{i:03d}").mkdir()

    _forget_sandbox_state(tools)
    assert Path(tools.resolve_sandbox_workdir(session)) == workdir
    assert Path(tools.get_sandbox_workdir(session)) == workdir
    assert tools.remove_session_sandbox(session, delete_files = True) is True


def test_a_staged_move_is_marked_before_the_rename(tmp_path, monkeypatch):
    """Across filesystems the move has already removed the legacy copy, so a
    kill before the marker leaves the only copy unfindable."""
    fake_home = tmp_path / "userprofile"
    session = "__LOCALID_kill111"
    legacy = fake_home / "studio_sandbox" / session
    legacy.mkdir(parents = True)
    (legacy / "thesis.txt").write_text("years of work", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    root = Path(tools.sandbox_root())
    root.mkdir(parents = True, exist_ok = True)

    class Killed(RuntimeError):
        pass

    real_rename = os.rename

    def killed_rename(src, dst):
        # Only the staging-to-target step: shutil.move uses rename to get there.
        if tools._STAGING_SUFFIX in str(dst):
            return real_rename(src, dst)
        raise Killed("the process went away here")

    monkeypatch.setattr(os, "rename", killed_rename)
    with pytest.raises(Killed):
        tools._staged_move(str(legacy), str(root / session), session)
    monkeypatch.undo()

    staged = [p for p in root.iterdir() if tools._STAGING_SUFFIX in p.name]
    assert staged, "the tree vanished"
    assert tools._marker_owner(str(staged[0])) == tools._sandbox_name(session)


def test_a_delete_finds_the_folder_this_run_made(tmp_path, monkeypatch):
    """A tool can remove the marker, and after that neither name resolves to the
    directory: the delete left it behind without even saying it kept anything."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    session = "__LOCALID_cache11"
    (root / tools._sandbox_name(session)).mkdir()  # the user's, so we fall back
    workdir = Path(tools.get_sandbox_workdir(session))
    (workdir / "report.csv").write_text("a,b\n", encoding = "utf-8")
    (workdir / tools._SANDBOX_MARKER).unlink()

    assert tools.session_sandbox_has_files(session) is True  # this run made it
    assert tools.remove_session_sandbox(session, delete_files = True) is True
    assert not workdir.exists(), "the folder was left behind with nobody able to reach it"


def test_a_chat_moved_out_of_a_project_survives_its_deletion():
    """The transaction decides membership, and an earlier listing would have
    stopped that chat's generation and deleted the files it wrote."""
    import inspect

    from routes import chat_history

    route = inspect.getsource(chat_history.delete_project)
    assert "list_chat_threads(project_id" not in route
    assert 'member_ids = list(project.get("memberIds") or [])' in route
    assert route.index("delete_chat_project(") < route.index("member_ids = list(")


def test_the_client_reads_the_file_line_only_from_the_sandbox_tools():
    """An MCP tool or a fetched page ending in that line is content, and the
    card it produced pointed at a file nobody wrote."""
    src = Path(__file__).resolve().parents[2] / "frontend/src"

    files = (src / "components/assistant-ui/sandbox-files.ts").read_text(encoding = "utf-8")
    assert "SANDBOX_FILE_TOOLS" in files
    assert '"python", "terminal"' in files

    adapter = (src / "features/chat/api/chat-adapter.ts").read_text(encoding = "utf-8")
    guarded = adapter[adapter.index("const rawEvent = (toolEvent.result as string)") :]
    guarded = guarded[: guarded.index("const imgMarker")]
    assert "SANDBOX_FILE_TOOLS.has(" in guarded
    assert guarded.index("SANDBOX_FILE_TOOLS.has(") < guarded.index("extractCreatedFiles(")


def test_the_old_shared_bucket_is_read_but_never_moved(tmp_path, monkeypatch):
    """Before this change every id the filesystem could not hold shared one
    directory, so it belongs to no single chat and must not travel as one."""
    fake_home = tmp_path / "userprofile"
    bucket = fake_home / "studio_sandbox" / "_invalid"
    bucket.mkdir(parents = True)
    (bucket / "old.csv").write_text("a,b\n", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False
    session = "client.v1"  # what the old regex rejected
    assert tools._usable_session_id(session) is False

    served = Path(tools.resolve_sandbox_workdir(session))
    assert (served / "old.csv").is_file(), f"the chat lost its files: {served}"

    tools._migrate_legacy_sandbox(tools.sandbox_root())
    assert (bucket / "old.csv").is_file(), "a shared bucket was moved as one chat's"
    assert not (Path(tools.sandbox_root()) / "_invalid").exists()
    # And still readable once the pass has run.
    assert (Path(tools.resolve_sandbox_workdir(session)) / "old.csv").is_file()


def test_a_chat_cannot_claim_another_chats_directory(tmp_path, monkeypatch):
    """Tool code runs inside the sandbox and can write anything into the marker,
    so adopting on the marker alone hands one chat another chat's files."""
    root = _shared_root(tmp_path, monkeypatch)

    from core.inference import tools

    _forget_sandbox_state(tools)
    attacker, victim = "__LOCALID_aaa1111", "__LOCALID_bbb2222"
    theirs = Path(tools.get_sandbox_workdir(attacker))
    (theirs / "private.csv").write_text("the attacker's own", encoding = "utf-8")
    # What a tool running in there can do.
    (theirs / tools._SANDBOX_MARKER).write_text(
        tools._sandbox_name(victim),
        encoding = "utf-8",
    )

    _forget_sandbox_state(tools)
    landed = Path(tools.get_sandbox_workdir(victim))
    assert landed != theirs, "one chat was handed another chat's directory"
    assert not (landed / "private.csv").exists()
    assert theirs.is_dir(), "and the other chat's folder was taken from it"


def test_a_users_own_marker_file_survives_the_migration(tmp_path, monkeypatch):
    """The name was not reserved before this change, so a chat that wrote its
    own .unsloth_sandbox has a real file there."""
    fake_home = tmp_path / "userprofile"
    session = "__LOCALID_marker1"
    legacy = fake_home / "studio_sandbox" / session
    legacy.mkdir(parents = True)
    (legacy / ".unsloth_sandbox").write_text("notes the user wrote", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools

    _forget_sandbox_state(tools)
    tools._legacy_sandbox_migrated = False

    landed = Path(tools.get_sandbox_workdir(session))
    saved = list(landed.glob(".unsloth_sandbox.saved*"))
    assert saved, sorted(p.name for p in landed.iterdir())
    assert saved[0].read_text(encoding = "utf-8") == "notes the user wrote"
    assert tools._marker_owner(str(landed)) == tools._sandbox_name(session)


def test_a_project_workspace_goes_after_its_tools_are_stopped():
    """The member chats' calls run with their cwd in there, and pulling it out
    from under a live subprocess strands whatever it writes next."""
    import inspect

    from routes import chat_history

    route = inspect.getsource(chat_history.delete_project)
    assert "delete_chat_project(project_id, delete_files = False)" in route
    assert route.index("_cancel_active_generations(member_ids)") < route.index(
        "delete_project_workspace"
    )


def test_a_forked_chat_keeps_the_files_its_cards_point_at(tmp_path, monkeypatch):
    """Forking clones the message content verbatim, so the fork's cards still
    name the source chat's sandbox."""
    import asyncio

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))

    from core.inference import tools
    from routes import chat_history

    _forget_sandbox_state(tools)
    source = "__LOCALID_source1"
    workdir = Path(tools.get_sandbox_workdir(source))
    (workdir / "report.csv").write_text("a,b\n", encoding = "utf-8")

    import storage.studio_db as studio_db

    monkeypatch.setattr(
        studio_db, "sandbox_is_referenced_elsewhere", lambda session_id: session_id == source
    )

    removed, kept = asyncio.new_event_loop().run_until_complete(
        chat_history._remove_sandboxes([source], True)
    )

    assert removed == 0, removed
    assert kept == [source], kept
    assert (workdir / "report.csv").is_file(), "a surviving fork's cards now 404"


def test_the_research_loop_keeps_a_pages_file_line():
    """The persisted excerpt is what a resumed run reads back, and a fetched
    page ending in that line is content, not an envelope."""
    import inspect

    from core import research_runs

    source = inspect.getsource(research_runs)
    assert "strip_result_for_model(result)" not in source
    assert source.count('strip_result_for_model(result, "web_search")') == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
