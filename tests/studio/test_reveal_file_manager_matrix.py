# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every platform branch of ``reveal_in_file_manager``, and how each one fails.

``test_reveal_file_manager.py`` beside this covers WSL and native Linux; this
completes the matrix with macOS, native Windows, and the failure modes once a
branch is chosen. Launchers are stubbed and argv asserted exactly, since what
breaks a path here is a space, comma or non-ASCII splitting one argument in two.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest


def _find_repo_root() -> Path | None:
    env = os.environ.get("UNSLOTH_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "studio" / "backend").is_dir():
            return p
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "studio" / "backend").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    pytest.skip(
        "Could not locate studio/backend. Set UNSLOTH_REPO_ROOT or run from "
        "the repository checkout.",
        allow_module_level = True,
    )

_STUDIO_BACKEND = _REPO_ROOT / "studio" / "backend"
if str(_STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(_STUDIO_BACKEND))

pytest.importorskip("fastapi")
pytest.importorskip("huggingface_hub")

try:
    from utils.paths import path_utils
except Exception as exc:
    pytest.skip(f"studio backend import unavailable: {exc}", allow_module_level = True)


# Characters that get split into two argv elements by string-built commands.
_AWKWARD_NAME = "rapport final, v2 (draft) — 90% ✅"


@pytest.fixture()
def spawned(monkeypatch):
    """Record what would have been launched, and launch nothing.

    ``startfile`` is Windows-only, so ``raising = False`` installs it rather
    than replacing it, which is what exercises that branch from Linux.
    """
    calls = types.SimpleNamespace(run = [], popen = [], startfile = [], popen_error = None)

    def fake_run(cmd, **kwargs):
        calls.run.append(list(cmd))
        return types.SimpleNamespace(stdout = "C:\\converted\n")

    def fake_popen(cmd, **kwargs):
        if calls.popen_error is not None:
            raise calls.popen_error
        calls.popen.append(list(cmd))
        return types.SimpleNamespace()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(os, "startfile", calls.startfile.append, raising = False)
    return calls


@pytest.fixture()
def macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "name", "posix")


@pytest.fixture()
def windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "name", "nt")


@pytest.fixture()
def native_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(path_utils, "_IS_WSL", False)


def test_macos_reveals_a_file_with_open_dash_r(macos, spawned, tmp_path):
    """``open -R`` selects the file in its enclosing folder; plain ``open``
    would hand the file to whichever application claims the extension."""
    target = tmp_path / "report.csv"
    target.write_text("a,b\n")
    path_utils.reveal_in_file_manager(target)
    assert spawned.popen == [["open", "-R", str(target)]]


def test_macos_opens_a_directory_rather_than_revealing_it(macos, spawned, tmp_path):
    """``open -R`` on a directory shows its PARENT with it selected, which for
    a sandbox is the root holding every other chat's."""
    path_utils.reveal_in_file_manager(tmp_path)
    assert spawned.popen == [["open", str(tmp_path)]]


def test_macos_keeps_an_awkward_name_in_one_argument(macos, spawned, tmp_path):
    target = tmp_path / _AWKWARD_NAME
    target.mkdir()
    path_utils.reveal_in_file_manager(target)
    assert spawned.popen == [["open", str(target)]]
    assert len(spawned.popen[0]) == 2


def test_windows_selects_a_file_in_explorer(windows, spawned, tmp_path):
    target = tmp_path / "report.csv"
    target.write_text("a,b\n")
    path_utils.reveal_in_file_manager(target)
    assert spawned.popen == [["explorer", f"/select,{target}"]]
    assert spawned.startfile == []


def test_windows_opens_a_directory_with_startfile(windows, spawned, tmp_path):
    path_utils.reveal_in_file_manager(tmp_path)
    assert spawned.startfile == [str(tmp_path)]
    assert spawned.popen == []


def test_windows_never_waits_on_explorer_or_reads_its_exit_code(windows, spawned, tmp_path):
    """``explorer.exe`` exits 1 even when it worked (microsoft/WSL#6565), so
    ``run(..., check = True)`` here would raise on every SUCCESSFUL reveal.
    ``Popen`` without a wait is what makes that harmless."""
    target = tmp_path / "report.csv"
    target.write_text("a,b\n")
    path_utils.reveal_in_file_manager(target)
    assert spawned.run == [], "explorer must not be run and waited on"
    assert spawned.popen and spawned.popen[0][0] == "explorer"


def test_windows_keeps_a_comma_in_the_path_out_of_the_select_flag(windows, spawned, tmp_path):
    """``/select,`` is comma-delimited and a filename may contain one. Recorded,
    not asserted-correct: it is one argv element, so parsing is Explorer's
    problem, not the shell's."""
    target = tmp_path / "q3, final.csv"
    target.write_text("a,b\n")
    path_utils.reveal_in_file_manager(target)
    assert spawned.popen == [["explorer", f"/select,{target}"]]
    assert len(spawned.popen[0]) == 2


def test_wsl_falls_back_to_xdg_open_when_wslpath_times_out(monkeypatch, spawned, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(path_utils, "_IS_WSL", True)

    def timing_out(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(subprocess, "run", timing_out)
    path_utils.reveal_in_file_manager(tmp_path)
    assert spawned.popen == [["xdg-open", str(tmp_path)]]


def test_wsl_falls_back_to_xdg_open_when_wslpath_fails(monkeypatch, spawned, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(path_utils, "_IS_WSL", True)

    def failing(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", failing)
    path_utils.reveal_in_file_manager(tmp_path)
    assert spawned.popen == [["xdg-open", str(tmp_path)]]


def test_native_linux_opens_a_directory_directly(native_linux, spawned, tmp_path):
    path_utils.reveal_in_file_manager(tmp_path)
    assert spawned.run == []
    assert spawned.popen == [["xdg-open", str(tmp_path)]]


def test_native_linux_keeps_an_awkward_name_in_one_argument(native_linux, spawned, tmp_path):
    target = tmp_path / _AWKWARD_NAME
    target.mkdir()
    path_utils.reveal_in_file_manager(target)
    assert spawned.popen == [["xdg-open", str(target)]]
    assert len(spawned.popen[0]) == 2


def test_a_deeply_nested_path_is_passed_through_whole(native_linux, spawned, tmp_path):
    """Sandbox names are derived, but a project workspace can sit arbitrarily
    deep under a home the user chose."""
    target = tmp_path.joinpath(*[f"level_{i}" for i in range(40)])
    target.mkdir(parents = True)
    path_utils.reveal_in_file_manager(target)
    assert spawned.popen == [["xdg-open", str(target)]]
    assert len(str(target)) > 255


@pytest.mark.parametrize("host", ["macos", "windows", "native_linux"])
def test_a_missing_target_launches_nothing_anywhere(host, spawned, tmp_path, request):
    request.getfixturevalue(host)
    with pytest.raises(FileNotFoundError):
        path_utils.reveal_in_file_manager(tmp_path / "never-existed")
    assert spawned.popen == []
    assert spawned.run == []
    assert spawned.startfile == []


@pytest.mark.parametrize("host", ["macos", "windows", "native_linux"])
def test_a_broken_symlink_launches_nothing_anywhere(host, spawned, tmp_path, request):
    """``exists()`` follows links, so a dangling one is "missing" -- which is
    the answer that matters: the old code would have opened its parent."""
    request.getfixturevalue(host)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "gone")
    with pytest.raises(FileNotFoundError):
        path_utils.reveal_in_file_manager(link)
    assert spawned.popen == []
    assert spawned.startfile == []


class _VanishesAfterTheGuard:
    """There for the existence guard, gone by the branch. The two are separate
    stats, and a chat deleted or migrated in between is what the endpoint's own
    tests already simulate one layer up."""

    def __init__(self, real: Path) -> None:
        self._real = real

    def exists(self) -> bool:
        return True

    def is_dir(self) -> bool:
        return False

    def is_file(self) -> bool:
        return False

    @property
    def parent(self) -> Path:
        return self._real.parent

    def __str__(self) -> str:
        return str(self._real)

    def __fspath__(self) -> str:
        return str(self._real)


def test_a_target_that_vanishes_after_the_guard_never_opens_its_parent(
    native_linux, spawned, tmp_path
):
    """The parent of a sandbox is the root holding every other chat's, so the
    Linux fallback must fail closed rather than widen to it."""
    root = tmp_path / "sandbox"
    (root / "thread-1").mkdir(parents = True)
    with pytest.raises(FileNotFoundError):
        path_utils.reveal_in_file_manager(_VanishesAfterTheGuard(root / "thread-1"))
    assert spawned.popen == [], "the sandbox root must never be opened"


class _SwappedForAFile:
    """A checked directory replaced by a regular file before the open. A tool
    runs inside its own sandbox and can do this, and the file branch names the
    parent on every platform."""

    def __init__(self, real: Path) -> None:
        self._real = real

    def exists(self) -> bool:
        return True

    def is_dir(self) -> bool:
        return False

    def is_file(self) -> bool:
        return True

    @property
    def parent(self) -> Path:
        return self._real.parent

    def __str__(self) -> str:
        return str(self._real)

    def __fspath__(self) -> str:
        return str(self._real)


@pytest.mark.parametrize("host", ["macos", "windows", "native_linux"])
def test_a_sandbox_swapped_for_a_file_is_refused_not_revealed(host, spawned, tmp_path, request):
    """``expect_dir`` is what the sandbox route passes, since a sandbox's parent
    is the root holding every other chat's. A real file on disk, not a fake:
    the one ``lstat`` leaves no gap between check and use for a fake to sit in,
    which is the property under test."""
    request.getfixturevalue(host)
    root = tmp_path / "sandbox"
    root.mkdir(parents = True)
    (root / "thread-1").write_bytes(b"not a directory any more")
    with pytest.raises(FileNotFoundError):
        path_utils.reveal_in_file_manager(root / "thread-1", expect_dir = True)
    assert spawned.popen == []
    assert spawned.startfile == []


@pytest.mark.parametrize("host", ["macos", "windows", "native_linux"])
def test_a_sandbox_swapped_for_a_directory_symlink_is_refused(host, spawned, tmp_path, request):
    """``is_dir()`` follows links, so a sandbox replaced by one would pass a
    naive check and open the TARGET. The guard asks for the link's own type."""
    request.getfixturevalue(host)
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    link = tmp_path / "sandbox" / "thread-1"
    link.parent.mkdir(parents = True)
    link.symlink_to(elsewhere, target_is_directory = True)
    assert link.is_dir(), "the premise: it looks like a directory"

    with pytest.raises(FileNotFoundError):
        path_utils.reveal_in_file_manager(link, expect_dir = True)
    assert spawned.popen == []
    assert spawned.startfile == []


@pytest.mark.parametrize("host", ["macos", "windows", "native_linux"])
def test_a_symlinked_cache_entry_is_still_revealed_without_expect_dir(
    host, spawned, tmp_path, request
):
    """A Hugging Face snapshot is a link farm, so the cached-model reveal must
    keep following links. Only the sandbox caller opts into the strict check."""
    request.getfixturevalue(host)
    blob = tmp_path / "blob.gguf"
    blob.write_bytes(b"gguf")
    link = tmp_path / "snapshot" / "model.gguf"
    link.parent.mkdir(parents = True)
    link.symlink_to(blob)
    path_utils.reveal_in_file_manager(link)
    assert spawned.popen or spawned.startfile


@pytest.mark.parametrize("host", ["macos", "windows", "native_linux"])
def test_the_same_swap_is_still_revealed_without_expect_dir(host, spawned, tmp_path, request):
    """The cached-model reveal points at a real file, so the flag is opt-in and
    the default behaviour is unchanged."""
    request.getfixturevalue(host)
    target = tmp_path / "model.gguf"
    path_utils.reveal_in_file_manager(_SwappedForAFile(target))
    assert spawned.popen or spawned.startfile


def test_a_missing_launcher_is_reported_as_a_missing_launcher(native_linux, spawned, tmp_path):
    """``xdg-open`` is absent on a headless host, and ``Popen`` then raises
    ``FileNotFoundError`` for the LAUNCHER just as for a missing target. The
    helper cannot tell them apart, so this pins the payload as the launcher,
    which is what lets the route answer 500 rather than "no folder"."""
    spawned.popen_error = FileNotFoundError(2, "No such file or directory", "xdg-open")
    with pytest.raises(FileNotFoundError) as caught:
        path_utils.reveal_in_file_manager(tmp_path)
    assert caught.value.filename == "xdg-open"
    assert tmp_path.is_dir(), "the folder was there the whole time"
