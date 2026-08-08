# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a chat's files live, and how a user gets them back.

Every assertion here failed before: the sandbox ignored UNSLOTH_STUDIO_HOME, only
images could be fetched, nothing listed a chat's files, bash reported none, the
compiled cache landed in the launcher's CWD, and a deleted chat left its folder
behind. Verified on Windows, macOS and Linux.
"""

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
    legacy.mkdir(parents=True)
    (legacy / "results.csv").write_text("a,b\n1,2\n")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising=False)

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
    assert sandbox_routes == [
        "/sandbox/{session_id}",
        "/sandbox/{session_id}/{filename}",
    ]

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
        ("py.csv", lambda: tools._python_exec(
            "open('py.csv','w').write('a,b\\n')", session_id=session)),
        ("sh.csv", lambda: tools._bash_exec(
            "printf 'a,b\\n' > sh.csv", session_id=session)),
    ):
        result = run()
        print(f"\n{name} -> {result!r}")
        assert "__FILES__:" in result, result
        assert name in result
        assert os.path.isfile(os.path.join(workdir, name))

    # The sentinel never reaches the model.
    from core.inference.tool_loop_controller import strip_result_for_model

    stripped = strip_result_for_model("done\n__FILES__:[{\"name\": \"x.csv\"}]")
    assert "__FILES__" not in stripped
    assert stripped.strip() == "done"


def test_internal_temp_files_are_not_reported(tmp_path, monkeypatch):
    """The executor's own scratch script is not a user-facing artifact."""
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    tools._workdirs.clear()
    result = tools._python_exec("print('hi')", session_id="__LOCALID_tmp999")
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
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising=False)

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
    configured.mkdir(parents=True)
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
    assert tools.remove_session_sandbox("__LOCALID_files22", delete_files=True) is True
    assert not withfile.exists()


def test_sandbox_removal_cannot_escape_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))

    from core.inference import tools

    outside = tmp_path / "precious"
    outside.mkdir()
    for bad in ("..", "../precious", "/etc", "project-abc", ""):
        assert tools.remove_session_sandbox(bad, delete_files=True) is False
    assert outside.is_dir()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
