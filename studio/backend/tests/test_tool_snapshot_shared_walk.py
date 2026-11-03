# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A python or terminal call with file history on walks its folder once per boundary, and that one
walk gives the files card and the history exactly what their own walks gave."""

import os
import random
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import file_history, tools
from core.inference.tools import execute_tool


def _reference_walk(workdir, history = False):
    """The single-view walk as it stood before views were shared, kept here as the oracle."""
    snapshot = {}
    if not workdir or not os.path.isdir(workdir):
        return snapshot
    max_files, max_dirs = (
        (tools._MAX_HISTORY_FILES, tools._MAX_HISTORY_DIRS)
        if history
        else (tools._MAX_SNAPSHOT_FILES, tools._MAX_SNAPSHOT_DIRS)
    )
    visited = 0
    hash_budget = 0 if tools._volume_timestamps_finely(workdir) else tools._MAX_SNAPSHOT_HASH_BYTES
    for base, dirs, names in os.walk(workdir):
        visited += 1
        if visited > max_dirs:
            return tools._CappedSnapshot(snapshot)
        folder = base[len(workdir) :].strip(os.sep)
        depth = len(tools._user_path_parts(folder.split(os.sep) if folder else []))
        prefix = folder.replace(os.sep, "/") + "/" if folder else ""
        dirs[:] = (
            []
            if not history and depth >= tools._MAX_SANDBOX_PATH_SEGMENTS - 1
            else [
                d
                for d in dirs
                if (not d.startswith(".") or (base == workdir and d == tools._ATTACHMENTS_DIR))
                and tools._servable_segment(d)
                and not tools._snapshot_skips_dir(base, d)
                and not (history and base == workdir and not file_history.tracked(d + "/", ()))
            ]
        )
        for name in names:
            if base == workdir and name in tools._INTERNAL_SANDBOX_FILES:
                continue
            if not tools._servable_segment(name):
                continue
            path = os.path.join(base, name)
            try:
                stat = os.lstat(path)
                if not tools.S_ISREG(stat.st_mode):
                    continue
                content = None
                if hash_budget and hash_budget >= stat.st_size:
                    content = tools._content_key(path, stat.st_size)
                    if content is not None:
                        hash_budget -= stat.st_size
                snapshot[prefix + name] = (stat.st_mtime_ns, stat.st_size, content)
            except OSError:
                continue
            if len(snapshot) >= max_files:
                return tools._CappedSnapshot(snapshot)
    return snapshot


_DIR_NAMES = [
    "a", "b", "src", "out", ".git", ".hidden", "node_modules", "__pycache__", "venv",
    "unsloth-tmp", ".unsloth_tool_output", tools._ATTACHMENTS_DIR, "deep",
]  # fmt: skip
_FILE_NAMES = ["x.txt", "y.py", ".gitignore", "data.csv", "big.bin", "z", tools._SANDBOX_MARKER]


def _random_tree(root: Path, rng: random.Random) -> None:
    def fill(folder: Path, depth: int) -> None:
        for name in rng.sample(_FILE_NAMES, rng.randint(0, len(_FILE_NAMES))):
            size = rng.choice([0, 3, 100, 5000])
            (folder / name).write_bytes(bytes(rng.getrandbits(8) for _ in range(size)))
        if depth >= 6:
            return
        for name in rng.sample(_DIR_NAMES, rng.randint(0, 4)):
            sub = folder / name
            sub.mkdir(exist_ok = True)
            if name == "venv":
                (sub / "pyvenv.cfg").write_text("home = x\n", encoding = "utf-8")
            fill(sub, depth + 1)
        if rng.random() < 0.3 and hasattr(os, "symlink"):
            try:
                os.symlink(str(folder), str(folder / "loop"))
                os.symlink(str(folder / "x.txt"), str(folder / "link.txt"))
            except OSError:
                pass

    fill(root, 0)


@pytest.mark.parametrize("seed", range(40))
def test_shared_walk_gives_each_view_its_own_walk(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    workdir = tmp_path / "w"
    workdir.mkdir()
    _random_tree(workdir, rng)
    monkeypatch.setattr(tools, "_MAX_SNAPSHOT_FILES", rng.choice([2000, 1, 5, 20]))
    monkeypatch.setattr(tools, "_MAX_SNAPSHOT_DIRS", rng.choice([2000, 1, 3, 10]))
    monkeypatch.setattr(tools, "_MAX_HISTORY_FILES", rng.choice([20000, 1, 7, 30]))
    monkeypatch.setattr(tools, "_MAX_HISTORY_DIRS", rng.choice([20000, 1, 4, 12]))
    monkeypatch.setattr(tools, "_MAX_SNAPSHOT_HASH_BYTES", rng.choice([64 << 20, 4000, 6000]))
    monkeypatch.setattr(tools, "_volume_timestamps_finely", lambda path, fine = seed % 2: bool(fine))

    for target in (str(workdir), str(tmp_path / "missing"), None):
        card, history = tools._snapshot_workdir_views(target, (False, True))
        for got, want in (
            (card, _reference_walk(target)),
            (history, _reference_walk(target, True)),
        ):
            assert got == want
            assert type(got) is type(want)
        assert tools._snapshot_workdir_files(target) == _reference_walk(target)
        assert type(tools._snapshot_workdir_files(target, history = True)) is type(
            _reference_walk(target, True)
        )


@pytest.fixture
def chat(tmp_path, monkeypatch):
    workdir = tmp_path / "sandbox"
    workdir.mkdir()
    history = file_history.FileHistory(str(tmp_path / "file_history"))
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    monkeypatch.setattr(file_history, "store", lambda: history)
    yield workdir, history
    file_history.set_turn(None, None)


@pytest.mark.parametrize("tool", ["python", "terminal"])
def test_a_call_with_history_lists_each_folder_once_per_boundary(chat, monkeypatch, tool):
    workdir, _ = chat
    for i in range(5):
        (workdir / f"d{i}").mkdir()
        (workdir / f"d{i}" / "f.txt").write_text("x\n", encoding = "utf-8")
    real_walk, listed = os.walk, []

    def counting_walk(top, *args, **kwargs):
        # tools.os is os: skip the sandbox's own pre-launch scan (seatbelt / bwrap), same on main.
        counted = sys._getframe(1).f_globals.get("__name__") == tools.__name__
        return _counted(real_walk(top, *args, **kwargs), counted)

    def _counted(walk, counted):
        for entry in walk:
            if counted and entry[0].startswith(str(workdir)):
                listed.append(entry[0])
            yield entry

    monkeypatch.setattr(tools.os, "walk", counting_walk)
    file_history.set_turn("chat-1", "u1")
    code = "open('out.txt', 'w').write('x')"
    # Forward slashes and quotes: the terminal tool runs bash on Windows too, which eats backslashes.
    _exe = sys.executable.replace(os.sep, "/")
    args = {"code": code} if tool == "python" else {"command": f'"{_exe}" -c "{code}"'}
    result = execute_tool(tool, args, session_id = "s")
    monkeypatch.setattr(tools.os, "walk", real_walk)

    assert "out.txt" in result
    for folder in [workdir] + [workdir / f"d{i}" for i in range(5)]:
        assert listed.count(str(folder)) == 2, (folder, listed)
    assert len(listed) == len(set(listed)) + 6, listed
