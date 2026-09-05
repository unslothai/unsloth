# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A Studio home whose venv is a symlink must not launch the CLI into a loop.

`unsloth studio run` hands off to the Studio venv's own console script when it is not
already running inside that venv. The check compared `sys.prefix`, which is the venv's REAL
directory, against `STUDIO_HOME / "unsloth_studio"` as a string prefix. With the venv
symlinked into the home (two homes sharing one venv, a relocated install) the child made
the same comparison, failed it the same way, and handed off to itself again: 100 percent
CPU, nothing printed, no server. Measured at over seven minutes before it was killed.

Two rules, pinned separately: the check resolves symlinks, and a second hand-off is refused
with a message rather than attempted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import typer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio
    return _studio


class TestTheVenvCheckResolvesSymlinks:
    def test_a_symlinked_venv_is_recognised(self, tmp_path, monkeypatch):
        real = tmp_path / "shared" / "unsloth_studio"
        (real / "bin").mkdir(parents = True)
        home = tmp_path / "home"
        home.mkdir()
        (home / "unsloth_studio").symlink_to(real, target_is_directory = True)
        monkeypatch.setattr(sys, "prefix", str(real))
        assert _studio()._running_inside_studio_venv(home / "unsloth_studio") is True, (
            "the interpreter IS the home's venv, reached through a symlink; saying "
            "otherwise hands off to a child that will say the same, forever"
        )

    def test_a_real_venv_is_still_recognised(self, tmp_path, monkeypatch):
        venv = tmp_path / "home" / "unsloth_studio"
        venv.mkdir(parents = True)
        monkeypatch.setattr(sys, "prefix", str(venv))
        assert _studio()._running_inside_studio_venv(venv) is True

    def test_another_interpreter_is_not(self, tmp_path, monkeypatch):
        venv = tmp_path / "home" / "unsloth_studio"
        venv.mkdir(parents = True)
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
        assert _studio()._running_inside_studio_venv(venv) is False

    def test_a_missing_venv_directory_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
        assert _studio()._running_inside_studio_venv(tmp_path / "nope" / "unsloth_studio") is False


class TestTheSecondHandOffIsRefused:
    def test_the_first_hand_off_marks_the_environment(self, monkeypatch):
        studio = _studio()
        # setenv, so the marker the guard writes is restored (to absent) at teardown and
        # cannot leak into the re-exec tests that run after this one.
        monkeypatch.setenv(studio._REEXEC_DEPTH_ENV, "0")
        studio._guard_reexec_loop("/some/home/unsloth_studio")
        assert os.environ.get(studio._REEXEC_DEPTH_ENV) == "1"

    def test_the_second_hand_off_exits_with_a_message(self, monkeypatch, capsys):
        studio = _studio()
        monkeypatch.setenv(studio._REEXEC_DEPTH_ENV, "1")
        with pytest.raises(typer.Exit) as excinfo:
            studio._guard_reexec_loop("/some/home/unsloth_studio")
        assert excinfo.value.exit_code == 2
        err = capsys.readouterr().err
        assert "Refusing to hand off again" in err
        assert "UNSLOTH_STUDIO_HOME" in err

    def test_garbage_in_the_marker_counts_as_a_first_hand_off(self, monkeypatch):
        studio = _studio()
        monkeypatch.setenv(studio._REEXEC_DEPTH_ENV, "not-a-number")
        studio._guard_reexec_loop("/some/home/unsloth_studio")
        assert os.environ.get(studio._REEXEC_DEPTH_ENV) == "1"
