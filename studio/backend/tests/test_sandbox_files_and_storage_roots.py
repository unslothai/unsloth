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
    for reserved in ("con", "NUL", "aux", "COM1", "lpt9", "nul.txt"):
        workdir = Path(tools.get_sandbox_workdir(reserved))
        assert workdir.name == "_invalid", reserved
        assert tools.remove_session_sandbox(reserved, delete_files = True) is False


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
    monkeypatch.setattr(chat_history, "clear_chat_history", lambda: None)
    monkeypatch.setattr(chat_history, "_cancel_active_research", lambda request, ids: None)

    import asyncio

    body = asyncio.run(chat_history.clear_history(request = None, current_subject = "tester"))
    assert body["sandboxes_removed"] == 1
    assert not empty.exists()


def test_the_legacy_migration_runs_for_a_read_too(tmp_path, monkeypatch):
    """Right after an upgrade, opening a file must not 404 until a tool runs."""
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
    resolved = tools.resolve_sandbox_workdir("__LOCALID_evil")
    assert Path(resolved).name == "_invalid", resolved
    assert not str(Path(resolved).resolve()).startswith(str(outside.resolve()))


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
    assert resolved.name == "_invalid", resolved
    assert not (resolved / "secret.txt").exists()

    executing = Path(tools.get_sandbox_workdir("__LOCALID_swapped"))
    assert executing.name == "_invalid", executing


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
    assert "delete_files = delete_files" in source


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

    real_rmdir = tools.os.rmdir

    def slow_rmdir(path):
        # Stand in for the window between deciding and unlinking.
        entered.set()
        time.sleep(0.3)
        return real_rmdir(path)

    monkeypatch.setattr(tools.os, "rmdir", slow_rmdir)
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
    """A training run has no sandbox, and a chat in a project shares the project
    workspace, which chat deletion does not touch."""
    sidebar = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "components" / "app-sidebar.tsx"
    ).read_text(encoding = "utf-8")
    assert "function deleteTargetHasFiles" in sidebar
    assert 'return target.kind === "chat" && !target.item.projectId;' in sidebar
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


def test_a_delete_interrupted_by_a_restart_is_finished_later(tmp_path, monkeypatch):
    """The detached directory is under a name no session id can reach, so an
    exit mid-delete would strand it for good."""
    import time

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_stale11"
    workdir = Path(tools.get_sandbox_workdir(session))
    root = workdir.parent

    # What a previous run left behind when it was killed mid-delete.
    stranded = root / f"__LOCALID_gone999{tools._DETACHED_SUFFIX}abc12345"
    stranded.mkdir()
    (stranded / tools._SANDBOX_MARKER).touch()
    (stranded / "leftover.bin").write_bytes(b"x")

    (workdir / "data.bin").write_bytes(b"y")
    assert tools.remove_session_sandbox(session, delete_files = True) is True
    for _ in range(50):
        if not stranded.exists() and not workdir.exists():
            break
        time.sleep(0.1)
    assert not workdir.exists()
    assert not stranded.exists(), "a previous run's leftover was never cleaned up"


def test_a_startup_sweep_finishes_an_interrupted_delete(tmp_path, monkeypatch):
    """The delete worker is a daemon, so an exit leaves the renamed directory
    with nothing to reclaim it until someone deletes another chat."""
    import time

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    root = Path(tools.sandbox_root())
    root.mkdir(parents = True, exist_ok = True)
    stranded = root / f"__LOCALID_gone111{tools._DETACHED_SUFFIX}deadbeef"
    stranded.mkdir()
    (stranded / tools._SANDBOX_MARKER).touch()
    (stranded / "leftover.bin").write_bytes(b"x")

    tools._detached_swept = False
    tools.sweep_detached_sandboxes()
    for _ in range(50):
        if not stranded.exists():
            break
        time.sleep(0.1)
    assert not stranded.exists(), "a delete interrupted by a restart was never finished"
    # Once per process.
    assert tools._detached_swept is True


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


def test_only_our_own_tombstones_are_swept(tmp_path, monkeypatch):
    """The root can be a folder the user already keeps things in, and a name
    merely containing the suffix is not one of ours."""
    root = tmp_path / "shared"
    root.mkdir()
    theirs = root / "report.deleting-backup"
    theirs.mkdir()
    (theirs / "notes.txt").write_text("keep me", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    import time

    from core.inference import tools

    ours = root / "__LOCALID_x.deleting-0a1b2c3d"
    ours.mkdir()
    (ours / tools._SANDBOX_MARKER).touch()
    # Named like one of ours, but nothing says we made it.
    lookalike = root / "archive.deleting-0a1b2c3d"
    lookalike.mkdir()
    (lookalike / "photos.zip").write_text("theirs", encoding = "utf-8")

    tools._detached_swept = False
    tools.sweep_detached_sandboxes()
    for _ in range(50):
        if not ours.exists():
            break
        time.sleep(0.05)
    assert not ours.exists(), "our own tombstone was left behind"
    assert (theirs / "notes.txt").read_text(encoding = "utf-8") == "keep me"
    assert (lookalike / "photos.zip").is_file(), "an unowned lookalike was deleted"


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


def test_a_half_finished_migration_is_finished_later(tmp_path, monkeypatch):
    """A cross-device move copies and then unlinks, so an interruption leaves
    files on both sides and the destination is not a collision."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox" / "__LOCALID_part111"
    legacy.mkdir(parents = True)
    (legacy / "left_behind.csv").write_text("rest", encoding = "utf-8")
    (legacy / "already_there.csv").write_text("older", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    root = Path(tools.sandbox_root())
    partial = root / "__LOCALID_part111"
    partial.mkdir(parents = True)
    # What the interrupted copy got through, and no claim: it never finished.
    (partial / "already_there.csv").write_text("newer", encoding = "utf-8")

    tools._migrate_legacy_sandbox(str(root))

    assert (partial / "left_behind.csv").read_text(encoding = "utf-8") == "rest"
    assert (partial / "already_there.csv").read_text(encoding = "utf-8") == "newer"
    # The one that could not move is still down there for the user.
    assert (legacy / "already_there.csv").is_file()
    assert not (legacy / "left_behind.csv").exists()


def test_a_partial_move_is_claimed_even_with_a_duplicate_left_below(tmp_path, monkeypatch):
    """A file that cannot move must not leave the destination reading as
    somebody else's, or the chat is sent to an empty directory instead."""
    fake_home = tmp_path / "userprofile"
    legacy = fake_home / "studio_sandbox" / "__LOCALID_dup222"
    legacy.mkdir(parents = True)
    (legacy / "kept.csv").write_text("older", encoding = "utf-8")
    (legacy / "moved.csv").write_text("rest", encoding = "utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    tools._legacy_sandbox_migrated = False
    partial = root / "__LOCALID_dup222"
    partial.mkdir()
    (partial / "kept.csv").write_text("newer", encoding = "utf-8")

    tools._migrate_legacy_sandbox(str(root))
    assert (partial / tools._SANDBOX_MARKER).is_file(), "the destination was left unclaimed"
    assert (partial / "moved.csv").read_text(encoding = "utf-8") == "rest"
    assert (legacy / "kept.csv").is_file(), "the conflict was not left for the user"
    # And the chat lands on the migrated files rather than a fresh directory.
    assert Path(tools.get_sandbox_workdir("__LOCALID_dup222")) == partial


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


def test_a_sandbox_we_cannot_claim_is_never_used(tmp_path, monkeypatch):
    """Running there would put this chat inside someone else's files."""
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(root))

    from core.inference import tools

    tools._workdirs.clear()
    session = "__LOCALID_taken1"
    for path in (root / session, Path(tools._disambiguated_session_dir(str(root), session))):
        path.mkdir()
        (path / tools._SANDBOX_MARKER).write_text("__LOCALID_other9", encoding = "utf-8")
        (path / "not_ours.txt").write_text("theirs", encoding = "utf-8")

    workdir = Path(tools._ensure_session_dir(str(root), session))
    assert not (workdir / "not_ours.txt").exists(), workdir
    assert (workdir / tools._SANDBOX_MARKER).read_text(encoding = "utf-8") == session


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
