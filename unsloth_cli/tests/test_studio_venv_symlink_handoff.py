# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A Studio home whose venv is a symlink must not launch the CLI into a loop."""

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
        # setenv, so the marker the guard writes is restored at teardown and cannot leak into
        # the re-exec tests that run after this one.
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


class TestTheMarkerIsClearedWhereTheHandOffLanded:
    def test_the_recognised_child_drops_the_marker(self, monkeypatch):
        """Left in place it reaches the server and every subprocess, and a fresh
        `unsloth studio` is refused as a second hand-off."""
        monkeypatch.setenv(_studio()._REEXEC_DEPTH_ENV, "1")
        _studio()._hand_off_landed()
        assert _studio()._REEXEC_DEPTH_ENV not in os.environ

    def test_both_commands_drop_it_on_the_in_venv_path(self):
        import inspect
        for command in (_studio().studio_default, _studio().run):
            source = inspect.getsource(command)
            landed = source.index("_hand_off_landed()")
            check = source.index("in_studio_venv = _running_inside_studio_venv(studio_venv_dir)")
            assert check < landed < check + 200

    def test_the_server_child_is_not_handed_the_marker(self):
        """`unsloth studio` execs run.py, the server itself, which never hands off again."""
        import inspect

        source = inspect.getsource(_studio().studio_default)
        pop = source.index("os.environ.pop(_REEXEC_DEPTH_ENV, None)")
        exec_at = source.index("os.execvp(str(studio_python), args)")
        assert pop < exec_at < pop + 200


class TestAnOldLauncherBehindASymlinkIsRefusedNotLooped:
    def _venv(self, tmp_path, *, symlinked, launcher_text):
        real = tmp_path / "real_venv"
        site = real / "lib" / "python3.12" / "site-packages" / "unsloth_cli" / "commands"
        site.mkdir(parents = True)
        (site / "studio.py").write_text(launcher_text, encoding = "utf-8")
        if not symlinked:
            return real
        link = tmp_path / "home" / "unsloth_studio"
        link.parent.mkdir(parents = True)
        link.symlink_to(real, target_is_directory = True)
        return link

    def test_an_old_launcher_behind_a_symlink_is_detected(self, tmp_path):
        venv = self._venv(tmp_path, symlinked = True, launcher_text = "def run(): pass\n")
        assert _studio()._child_launcher_predates_the_guard(venv) is True

    def test_a_launcher_that_reads_the_marker_is_fine(self, tmp_path):
        venv = self._venv(
            tmp_path,
            symlinked = True,
            launcher_text = f'{_studio()._REEXEC_DEPTH_ENV} = "x"\n',
        )
        assert _studio()._child_launcher_predates_the_guard(venv) is False

    def test_a_real_venv_is_never_refused(self, tmp_path):
        """Without a symlink the old prefix check passes as it always did."""
        venv = self._venv(tmp_path, symlinked = False, launcher_text = "def run(): pass\n")
        assert _studio()._child_launcher_predates_the_guard(venv) is False

    def test_a_venv_whose_launcher_cannot_be_found_is_given_the_benefit_of_the_doubt(
        self, tmp_path
    ):
        real = tmp_path / "real_venv"
        real.mkdir()
        link = tmp_path / "unsloth_studio"
        link.symlink_to(real, target_is_directory = True)
        assert _studio()._child_launcher_predates_the_guard(link) is False

    def test_the_refusal_names_the_venv_and_exits_2(self, tmp_path, capsys):
        venv = self._venv(tmp_path, symlinked = True, launcher_text = "def run(): pass\n")
        with pytest.raises(typer.Exit) as raised:
            _studio()._refuse_an_old_launcher_behind_a_symlink(venv, "/x/bin/python")
        assert raised.value.exit_code == 2
        err = capsys.readouterr().err
        assert "symlink" in err and str(venv) in err

    def test_run_asks_before_it_hands_off(self):
        import inspect

        source = inspect.getsource(_studio().run)
        ask = source.index(
            "_refuse_an_old_launcher_behind_a_symlink(studio_venv_dir, studio_python)"
        )
        guard = source.index("_guard_reexec_loop(str(studio_venv_dir))")
        assert ask < guard

    def test_both_guards_run_before_the_windows_branch(self):
        """Windows hands off with subprocess.Popen, which inherits this environment just as
        os.execvp does. Guarding only the POSIX arm let an old child behind a symlinked venv
        spawn another child, and another, with no depth marker anywhere to stop it."""
        import inspect

        source = inspect.getsource(_studio().run)
        ask = source.index(
            "_refuse_an_old_launcher_behind_a_symlink(studio_venv_dir, studio_python)"
        )
        guard = source.index("_guard_reexec_loop(str(studio_venv_dir))")
        branch = source.index('if sys.platform == "win32":')
        spawn = source.index("subprocess.Popen(args")
        assert ask < branch and guard < branch, (
            "the platform branch is taken before the hand-off guards, so the Windows "
            "child is started unguarded"
        )
        assert guard < spawn
