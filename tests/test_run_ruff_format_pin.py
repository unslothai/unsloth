# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The formatter must run the ruff the repo is formatted with, or refuse.

ruff's output is not stable across releases: 0.9 changed which half of an
`assert cond, "msg"` gets wrapped. So a contributor whose environment has a newer
ruff than `.pre-commit-config.yaml` pins produces files the hook reformats back,
and pre-commit.ci fails the PR on files nobody broke -- which is how four files
reached main in that state, and how two PRs went red without a code defect
between them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from run_ruff_format import (  # noqa: E402
    ANY_VERSION_ENV,
    CONFIG,
    installed_ruff_version,
    main,
    pinned_ruff_version,
    version_mismatch,
)


class TestReadingThePin:
    def test_the_repo_pins_a_ruff_and_it_is_readable(self):
        # The pin is the contract this whole check rests on. If the hook stops
        # naming a version, or names it another way, this is where that shows up.
        assert CONFIG.exists()
        assert pinned_ruff_version(CONFIG.read_text(encoding = "utf-8"))

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("        additional_dependencies:\n          - ruff==0.6.9\n", "0.6.9"),
            ("  - ruff == 0.6.9\n", "0.6.9"),
            ("  - ruff==0.6.9  # keep in step with the formatter\n", "0.6.9"),
            ("  - ruff==0.12.0rc1\n", "0.12.0rc1"),
        ],
    )
    def test_the_spellings_a_config_may_use(self, text, expected):
        assert pinned_ruff_version(text) == expected

    def test_no_pin_is_not_a_mismatch(self):
        # An unpinned hook is a different problem, and refusing to format would be
        # the wrong answer to it.
        assert pinned_ruff_version("repos:\n  - repo: local\n") is None
        assert version_mismatch(None, "0.16.6") is False

    def test_two_different_pins_answer_nothing(self):
        # Which one would we enforce? Neither: say nothing rather than guess.
        assert pinned_ruff_version("  - ruff==0.6.9\n  - ruff==0.9.0\n") is None

    def test_the_same_pin_twice_is_still_one_answer(self):
        assert pinned_ruff_version("  - ruff==0.6.9\n  - ruff==0.6.9\n") == "0.6.9"


class TestTheMismatchRule:
    def test_a_newer_ruff_is_a_mismatch(self):
        assert version_mismatch("0.6.9", "0.16.6") is True

    def test_the_pinned_ruff_is_not(self):
        assert version_mismatch("0.6.9", "0.6.9") is False

    def test_an_unreadable_ruff_is_not(self):
        # `python -m ruff --version` failing means the format below fails too, and
        # loudly. Refusing here would only replace one error with a worse one.
        assert version_mismatch("0.6.9", None) is False


class TestRefusing:
    @staticmethod
    def _fake_ruff(tmp_path: Path, version: str) -> str:
        """A python whose `-m ruff --version` answers `version` and formats nothing."""
        pkg = tmp_path / "ruff"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            f"import sys\nprint('ruff {version}')\nsys.exit(0)\n",
            encoding = "utf-8",
        )
        shim = tmp_path / "python_shim.py"
        shim.write_text(
            "import runpy, sys\n"
            f"sys.path.insert(0, {str(tmp_path)!r})\n"
            "runpy.run_module('ruff', run_name='__main__')\n",
            encoding = "utf-8",
        )
        return str(shim)

    def test_a_mismatched_ruff_is_reported_by_version(self, tmp_path):
        # The message has to name both versions: "reformatted by the hook" on its
        # own sends people looking for a defect in their diff.
        shim = self._fake_ruff(tmp_path, "9.9.9")
        assert installed_ruff_version(sys.executable) != "9.9.9"
        out = subprocess.run([sys.executable, shim], capture_output = True, text = True, timeout = 60)
        assert "ruff 9.9.9" in out.stdout

    def test_it_refuses_before_touching_a_file(self, tmp_path, monkeypatch):
        # The refusal must come first. A run that rewrites half the argument list
        # and then declines is worse than either outcome.
        target = tmp_path / "sample.py"
        original = "x = f(a = 1)\n"
        target.write_text(original, encoding = "utf-8")
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        monkeypatch.delenv(ANY_VERSION_ENV, raising = False)
        assert main([str(target)]) == 1
        assert target.read_text(encoding = "utf-8") == original

    @pytest.mark.skipif(
        installed_ruff_version() is None,
        reason = "this one really runs the formatter, and ruff is not installed here",
    )
    def test_the_override_lets_it_through(self, tmp_path, monkeypatch):
        # An escape hatch, because a pin bump has to be runnable before it is merged.
        #
        # Skipped rather than asserted where ruff is absent: main() forwards the
        # formatter's own exit code, so on an interpreter with no ruff this returns
        # 1 for a reason that has nothing to do with the version gate. That is how
        # it failed on the repo-tests CI runner, which installs no ruff.
        target = tmp_path / "sample.py"
        target.write_text("x = f(a=1)\n", encoding = "utf-8")
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        monkeypatch.setenv(ANY_VERSION_ENV, "1")
        assert main([str(target)]) == 0
        # And it really ran: the post-pass is what puts the spaces in.
        assert target.read_text(encoding = "utf-8") == "x = f(a = 1)\n"

    def test_no_files_is_still_a_no_op(self, monkeypatch):
        # pre-commit calls with the changed files; an unrelated commit passes none,
        # and must not be failed for a version it never used.
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        assert main([]) == 0


class TestTheScriptStaysRunnable:
    """It is invoked as a program, so the mode bit matters, and a wholesale rewrite drops it invisibly."""

    @pytest.mark.skipif(sys.platform.startswith("win"), reason = "no POSIX mode bits")
    def test_the_formatter_is_executable(self):
        script = _ROOT / "scripts" / "run_ruff_format.py"
        assert script.read_text(encoding = "utf-8").startswith("#!")
        assert (
            script.stat().st_mode & 0o111
        ), "scripts/run_ruff_format.py lost its executable bit; git tracks it as 100755"
