# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import shutil
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.tools as tools

_PATHEXT = ".EXE;.CMD;.BAT;.COM".replace(";", os.pathsep)


def _eager_reference() -> tuple[str, str]:
    """The pre-change resolver, verbatim: full candidate list first, then the first trusted one."""
    exts = [e for e in (os.environ.get("PATHEXT") or ".EXE;.CMD;.BAT;.COM").split(os.pathsep)]
    candidates: list[str] = []
    primary = shutil.which("git")
    if primary:
        candidates.append(primary)
    for entry in (os.environ.get("PATH") or "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if not entry or not os.path.isabs(entry):
            continue
        for ext in exts:
            cand = os.path.join(entry, "git" + ext)
            if os.path.isfile(cand):
                candidates.append(cand)
    for git_exe in candidates:
        git_dir = os.path.dirname(git_exe)
        if os.path.isabs(git_dir) and tools._is_trusted_windows_program_dir(git_dir):
            return os.path.realpath(git_dir), os.path.splitext(git_exe)[1].upper()
    return "", ""


def _layout(
    tmp_path,
    *,
    which_hit,
    path_dirs,
    filler = 40,
):
    """Program Files root plus PATH; ``path_dirs`` is a list of (dir, has_git) under tmp_path."""
    prog = tmp_path / "Program Files"
    prog.mkdir(exist_ok = True)
    entries = []
    for rel, has_git in path_dirs:
        d = tmp_path / rel
        d.mkdir(parents = True, exist_ok = True)
        if has_git:
            (d / "git.EXE").write_bytes(b"")
        entries.append(str(d))
    for i in range(filler):
        d = tmp_path / f"filler{i}"
        d.mkdir(exist_ok = True)
        entries.append(str(d))
    which = str(tmp_path / which_hit / "git.EXE") if which_hit else None
    return prog, os.pathsep.join(entries), which


SCENARIOS = {
    "which_trusted": dict(
        which_hit = "Program Files/Git/cmd",
        path_dirs = [("Program Files/Git/cmd", True), ("scoop/shims", True)],
    ),
    "which_untrusted_then_trusted": dict(
        which_hit = "scoop/shims",
        path_dirs = [("scoop/shims", True), ("Program Files/Git/cmd", True)],
    ),
    "untrusted_only": dict(
        which_hit = "scoop/shims",
        path_dirs = [("scoop/shims", True), ("Users/me/bin", True)],
    ),
    "nothing": dict(which_hit = None, path_dirs = [("Users/me/bin", False)]),
}


@pytest.fixture
def win_env(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PATHEXT", _PATHEXT)

    def _apply(prog, path, which):
        monkeypatch.setattr(tools, "_windows_program_roots", lambda: [str(prog)])
        monkeypatch.setenv("PATH", path)
        monkeypatch.setattr(tools.shutil, "which", lambda name: which if name == "git" else None)

    return _apply


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_lazy_scan_matches_eager_reference(name, win_env, tmp_path):
    prog, path, which = _layout(tmp_path, **SCENARIOS[name])
    win_env(prog, path, which)
    got = tools._resolve_trusted_windows_git()
    assert got == _eager_reference()
    if name in ("which_trusted", "which_untrusted_then_trusted"):
        assert got[0] == os.path.realpath(str(tmp_path / "Program Files" / "Git" / "cmd"))
        assert got[1] == ".EXE"
    else:
        assert got == ("", "")


def _count_isfile(monkeypatch):
    calls = []
    real = os.path.isfile

    def _isfile(p):
        calls.append(p)
        return real(p)

    monkeypatch.setattr(tools.os.path, "isfile", _isfile)
    return calls


def test_trusted_which_hit_skips_the_path_scan(win_env, tmp_path, monkeypatch):
    prog, path, which = _layout(tmp_path, **SCENARIOS["which_trusted"])
    win_env(prog, path, which)
    calls = _count_isfile(monkeypatch)
    assert tools._resolve_trusted_windows_git()[0]
    assert calls == []


def test_trusted_path_entry_stops_the_scan(win_env, tmp_path, monkeypatch):
    prog, path, which = _layout(tmp_path, **SCENARIOS["which_untrusted_then_trusted"])
    win_env(prog, path, which)
    calls = _count_isfile(monkeypatch)
    assert tools._resolve_trusted_windows_git()[0]
    assert len(calls) <= 2 * len(_PATHEXT.split(os.pathsep))
    assert not any("filler" in c for c in calls)


def test_untrusted_only_still_scans_everything(win_env, tmp_path, monkeypatch):
    prog, path, which = _layout(tmp_path, **SCENARIOS["untrusted_only"])
    win_env(prog, path, which)
    calls = _count_isfile(monkeypatch)
    assert tools._resolve_trusted_windows_git() == ("", "")
    assert len(calls) == len(path.split(os.pathsep)) * len(_PATHEXT.split(os.pathsep))


def test_trust_is_rechecked_every_call(win_env, tmp_path, monkeypatch):
    prog, path, which = _layout(tmp_path, **SCENARIOS["which_trusted"])
    win_env(prog, path, which)
    assert tools._resolve_trusted_windows_git()[0]
    monkeypatch.setattr(tools, "_windows_program_roots", lambda: [])
    assert tools._resolve_trusted_windows_git() == ("", "")
