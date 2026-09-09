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
    parse_files,
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

    def test_no_files_is_rejected_before_the_version_is_consulted(self, monkeypatch):
        # This used to assert `main([]) == 0`, on the premise that an unrelated
        # commit calls the hook with no files. It does not: the hook is
        # `types: [python]` without `pass_filenames: false`, so pre-commit skips
        # it outright when nothing matches rather than running it empty. The
        # silent success only ever reached humans, and told them the formatter
        # was a fixed point when it had not run.
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        assert main([]) == 2


class TestArgumentsAreTakenSeriously:
    """Every argument is a file to rewrite, so anything else has to be an error.

    The old handling kept `[arg for arg in argv if Path(arg).exists()]` and
    dropped the rest without a word, which had three faces: no arguments exited
    0 having formatted nothing, a typo'd path formatted nothing just as quietly,
    and `--check FILE` lost the flag, kept the file, and wrote to it.
    """

    def test_an_unknown_flag_is_refused_and_nothing_is_written(self, tmp_path, monkeypatch):
        # The one that did damage: the flag was dropped, the file behind it was
        # not, and a run asked to check rewrote the file instead.
        target = tmp_path / "sample.py"
        original = "x = f(a=1)\n"
        target.write_text(original, encoding = "utf-8")
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        assert main(["--check", str(target)]) == 2
        assert target.read_text(encoding = "utf-8") == original

    def test_check_says_what_the_script_actually_does(self):
        # "unsupported option" alone invites a retry without the flag, which is
        # the write the caller was trying to avoid.
        _, error = parse_files(["--check", "any.py"])
        assert error is not None
        assert "--check" in error
        assert "always rewrites" in error

    def test_it_does_not_offer_ruff_as_a_read_only_equivalent(self):
        # This script is enforce_kwargs_spacing --pre, then ruff, then
        # enforce_kwargs_spacing again. `ruff format --check` covers the middle
        # pass only, so a file can pass it cleanly and still be rewritten here.
        # Sending people there would rebuild, in the error message, the same
        # false green the argument handling above was fixed to stop producing.
        _, error = parse_files(["--check", "any.py"])
        assert error is not None
        lowered = error.lower()
        # Naming ruff is fine. Presenting it as the substitute is not, so if it
        # is named it has to be disclaimed in the same breath.
        if "ruff" in lowered:
            assert "not an equivalent" in lowered
        # And an honest alternative is offered rather than a partial one.
        assert "diff" in lowered

    @pytest.mark.parametrize("flag", ["-q", "--diff", "--fix", "--unknown"])
    def test_options_are_refused_by_shape_not_by_a_list(self, flag):
        _, error = parse_files([flag])
        assert error is not None and flag in error

    def test_a_missing_path_is_named(self, tmp_path):
        # Previously this formatted nothing and exited 0, so a typo looked like a
        # successful run over the file you meant.
        missing = tmp_path / "typo.py"
        files, error = parse_files([str(missing)])
        assert files == []
        assert error is not None and str(missing) in error

    def test_a_missing_path_fails_even_beside_a_real_one(self, tmp_path, monkeypatch):
        # Partial credit is the whole bug: formatting the file that exists and
        # ignoring the one that does not still reports success.
        real = tmp_path / "real.py"
        real.write_text("x = 1\n", encoding = "utf-8")
        missing = tmp_path / "nope.py"
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        assert main([str(real), str(missing)]) == 2

    def test_real_files_still_go_through_untouched_by_the_new_check(self, tmp_path):
        files, error = parse_files([str(tmp_path)])
        assert error is None
        assert files == [str(tmp_path)]


class TestTheScriptStaysRunnable:
    """The shebang and the mode bit are kept so it still runs as a program.

    The hook no longer depends on them -- its entry is `python scripts/...` now,
    because an autofix commit dropping the bit broke main twice -- but people do
    run it directly, and a wholesale rewrite drops the bit invisibly.
    """

    @pytest.mark.skipif(sys.platform.startswith("win"), reason = "no POSIX mode bits")
    def test_the_formatter_is_executable(self):
        script = _ROOT / "scripts" / "run_ruff_format.py"
        assert script.read_text(encoding = "utf-8").startswith("#!")
        assert (
            script.stat().st_mode & 0o111
        ), "scripts/run_ruff_format.py lost its executable bit; git tracks it as 100755"
