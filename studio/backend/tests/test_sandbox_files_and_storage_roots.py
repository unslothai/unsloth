# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a chat's files live, and how a user gets them back.

Every assertion here failed before: the sandbox ignored UNSLOTH_STUDIO_HOME, only
images could be fetched, nothing listed a chat's files, bash reported none, the
compiled cache landed in the launcher's CWD, and a deleted chat left its folder
behind. Verified on Windows, macOS and Linux.
"""

import json
import os
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


def test_leftover_executor_scratch_does_not_leak_the_folder(tmp_path, monkeypatch):
    """A tool call that just finished can leave studio_exec_ behind, and rmdir
    would then keep the folder forever."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_scratch"))
    (workdir / "studio_exec_abc123.py").write_text("print(1)")

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
    """Only the generated studio_exec_<random>.py is internal; a tool may
    legitimately write studio_exec_results.csv."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    workdir = Path(tools.get_sandbox_workdir("__LOCALID_prefix1"))
    (workdir / "studio_exec_results.csv").write_text("a,b\n")
    (workdir / "studio_exec_ab12cd.py").write_text("print(1)")

    # Reported, because it is the user's.
    assert "studio_exec_results.csv" in tools._snapshot_workdir_files(str(workdir))
    # And it blocks removal without the opt-in, unlike the generated script.
    assert tools.remove_session_sandbox("__LOCALID_prefix1") is False
    assert (workdir / "studio_exec_results.csv").is_file()
    assert not (workdir / "studio_exec_ab12cd.py").exists()


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
    assert not cache.exists()


def test_generated_modules_identify_a_cache_without_a_marker(tmp_path, monkeypatch):
    """An install that predates the marker is still cleaned, file by file."""
    cache = tmp_path / "old_cache"
    cache.mkdir()
    (cache / "unsloth_compiled_module_gemma3.py").write_text("x = 1\n")
    (cache / "UnslothSFTTrainer.py").write_text("trainer\n")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(cache))

    from utils import cache_cleanup

    cache_cleanup.clear_unsloth_compiled_cache()
    assert not (cache / "unsloth_compiled_module_gemma3.py").exists()
    assert not (cache / "UnslothSFTTrainer.py").exists()


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
    workdir.rmdir()
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
    assert not cache.exists()


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
    """studio_exec_results.csv is the user's; only studio_exec_<token>.py is ours,
    and the snapshot and download route already agree on that."""
    from routes import inference

    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    (sandbox / "studio_exec_results.csv").write_text("a,b\n")
    (sandbox / "studio_exec_ab12cd.py").write_text("print(1)")

    names = inference._sandbox_listing_names(str(sandbox))
    assert names == ["studio_exec_results.csv"], names


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
    (root / "__LOCALID_dupe123").mkdir(parents = True)

    tools._migrate_legacy_sandbox(str(root))
    assert tools._legacy_sandbox_migrated is True
    assert (legacy / "old.csv").is_file(), "the legacy copy was not left behind"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
