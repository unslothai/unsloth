# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Use the ruff-unsloth .pre-commit-config.yaml pins or refuse; its output changes between releases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import run_ruff_format  # noqa: E402
from run_ruff_format import (  # noqa: E402
    ANY_VERSION_ENV,
    CONFIG,
    installed_ruff_version,
    main,
    parse_files,
    pinned_requirements,
    pinned_ruff_version,
    version_mismatch,
)

_URL = "https://github.com/unslothai/ruff/releases/download/v0.6.9-unsloth.1/"
_LINUX = (
    f'"ruff-unsloth @ {_URL}ruff_unsloth-0.6.9.post1-py3-none-manylinux_2_17_x86_64.whl'
    "#sha256=00 ; sys_platform == 'linux'\""
)
_MAC = (
    f'"ruff-unsloth @ {_URL}ruff_unsloth-0.6.9.post1-py3-none-macosx_11_0_arm64.whl'
    "#sha256=11 ; sys_platform == 'darwin'\""
)


class TestReadingThePin:
    def test_the_repo_pins_a_ruff_and_it_is_readable(self):
        assert CONFIG.exists()
        assert pinned_ruff_version(CONFIG.read_text(encoding = "utf-8"))

    def test_the_repo_lists_a_wheel_per_platform(self):
        requirements = pinned_requirements(CONFIG.read_text(encoding = "utf-8"))
        assert len(requirements) >= 2
        assert all("#sha256=" in r and " ; " in r for r in requirements), requirements

    @pytest.mark.parametrize(
        "text, expected",
        [
            (f"        additional_dependencies:\n          - {_LINUX}\n", "0.6.9.post1"),
            (f"  - {_LINUX}\n  - {_MAC}\n", "0.6.9.post1"),
            (f"  - {_LINUX}  # keep in step with the formatter\n", "0.6.9.post1"),
            ("  - ruff-unsloth==0.6.9.post2\n", "0.6.9.post2"),
            ("  - ruff-unsloth == 0.6.9.post2 ; sys_platform == 'linux'\n", "0.6.9.post2"),
        ],
    )
    def test_the_spellings_a_config_may_use(self, text, expected):
        assert pinned_ruff_version(text) == expected

    def test_the_requirement_lines_come_back_installable(self):
        # The quotes are YAML's, not part of the requirement pip is handed.
        assert pinned_requirements(f"  - {_LINUX}\n") == [_LINUX.strip('"')]

    def test_stock_ruff_is_not_the_pin(self):
        # The lint hook's ruff is a different tool; only ruff-unsloth formats.
        assert pinned_ruff_version("  - ruff==0.6.9\n") is None

    def test_no_pin_is_not_a_mismatch(self):
        assert pinned_ruff_version("repos:\n  - repo: local\n") is None
        assert version_mismatch(None, "0.16.6") is False

    def test_two_different_pins_answer_nothing(self):
        text = f"  - {_LINUX}\n  - ruff-unsloth==0.6.9.post2\n"
        assert pinned_ruff_version(text) is None


class TestTheMismatchRule:
    def test_a_newer_ruff_is_a_mismatch(self):
        assert version_mismatch("0.6.9", "0.16.6") is True

    def test_the_pinned_ruff_is_not(self):
        assert version_mismatch("0.6.9", "0.6.9") is False

    def test_an_unreadable_version_string_is_not(self):
        # An unparsable version is not a mismatch; a ruff that does not run is refused earlier.
        assert version_mismatch("0.6.9", None) is False


class TestRuffHasToBeRunnableFirst:
    """A missing or broken ruff must refuse before the pre-pass strips magic commas from any file."""

    def test_a_binary_that_is_not_there_is_reported_not_raised(self, tmp_path):
        reason = run_ruff_format.ruff_unavailable_reason(str(tmp_path / "no-such-python"))
        assert reason is not None
        assert "Error" in reason or "error" in reason

    def test_a_ruff_that_exits_nonzero_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            run_ruff_format.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "No module named ruff"),
        )
        assert run_ruff_format.ruff_unavailable_reason() == "No module named ruff"

    def test_a_silent_failure_still_says_something(self, monkeypatch):
        monkeypatch.setattr(
            run_ruff_format.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 3, "", ""),
        )
        reason = run_ruff_format.ruff_unavailable_reason()
        assert reason and "3" in reason

    def test_a_working_ruff_is_available(self, monkeypatch):
        monkeypatch.setattr(
            run_ruff_format.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "ruff 0.6.9\n", ""),
        )
        assert run_ruff_format.ruff_unavailable_reason() is None

    def test_the_magic_commas_survive_a_run_with_no_ruff(self, tmp_path, monkeypatch):
        target = tmp_path / "kernel.py"
        original = "def f(\n    a,\n    b,\n):\n    return a\n"
        target.write_text(original, encoding = "utf-8")
        monkeypatch.setattr(
            "run_ruff_format.ruff_unavailable_reason", lambda *a, **k: "No module named ruff"
        )
        assert main([str(target)]) == 1
        assert target.read_text(encoding = "utf-8") == original

    def test_the_override_does_not_buy_a_run_without_ruff(self, tmp_path, monkeypatch):
        # ANY_VERSION_ENV waives a wrong ruff version, not a missing ruff.
        target = tmp_path / "sample.py"
        original = "x = f(a=1)\n"
        target.write_text(original, encoding = "utf-8")
        monkeypatch.setattr(
            "run_ruff_format.ruff_unavailable_reason", lambda *a, **k: "No module named ruff"
        )
        monkeypatch.setenv(ANY_VERSION_ENV, "1")
        assert main([str(target)]) == 1
        assert target.read_text(encoding = "utf-8") == original

    def test_the_message_names_the_pin_to_install(self, tmp_path, monkeypatch, capsys):
        target = tmp_path / "sample.py"
        target.write_text("x = 1\n", encoding = "utf-8")
        monkeypatch.setattr(
            "run_ruff_format.ruff_unavailable_reason", lambda *a, **k: "No module named ruff"
        )
        assert main([str(target)]) == 1
        err = capsys.readouterr().err
        assert "ruff-unsloth" in err and "additional_dependencies" in err
        assert pinned_ruff_version(CONFIG.read_text(encoding = "utf-8")) in err


class TestRefusing:
    @staticmethod
    def _fake_ruff(tmp_path: Path, version: str) -> str:
        """A python whose `-m ruff_unsloth --version` answers `version` and formats nothing."""
        pkg = tmp_path / "ruff_unsloth"
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
            "runpy.run_module('ruff_unsloth', run_name='__main__')\n",
            encoding = "utf-8",
        )
        return str(shim)

    def test_a_mismatched_ruff_is_reported_by_version(self, tmp_path):
        shim = self._fake_ruff(tmp_path, "9.9.9")
        assert installed_ruff_version(sys.executable) != "9.9.9"
        out = subprocess.run([sys.executable, shim], capture_output = True, text = True, timeout = 60)
        assert "ruff 9.9.9" in out.stdout

    def test_it_refuses_before_touching_a_file(self, tmp_path, monkeypatch):
        # The refusal must come before any rewriting.
        target = tmp_path / "sample.py"
        original = "x = f(a = 1)\n"
        target.write_text(original, encoding = "utf-8")
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        monkeypatch.delenv(ANY_VERSION_ENV, raising = False)
        assert main([str(target)]) == 1
        assert target.read_text(encoding = "utf-8") == original

    @pytest.mark.skipif(
        installed_ruff_version() is None,
        reason = "this one really runs the formatter, and ruff-unsloth is not installed here",
    )
    def test_the_override_lets_it_through(self, tmp_path, monkeypatch):
        # Skipped without ruff: main() forwards the formatter's exit code, unrelated to the gate.
        target = tmp_path / "sample.py"
        target.write_text("x = f(a=1)\n", encoding = "utf-8")
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        monkeypatch.setenv(ANY_VERSION_ENV, "1")
        assert main([str(target)]) == 0
        # ruff-unsloth is what puts the spaces in, so this proves it ran.
        assert target.read_text(encoding = "utf-8") == "x = f(a = 1)\n"

    def test_no_files_is_rejected_before_the_version_is_consulted(self, monkeypatch):
        # pre-commit skips this hook when no python files match, so it is never run empty.
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        assert main([]) == 2


class TestArgumentsAreTakenSeriously:
    """Every argument must be a file to rewrite; an unknown flag or missing path is an error, not
    dropped."""

    def test_an_unknown_flag_is_refused_and_nothing_is_written(self, tmp_path, monkeypatch):
        target = tmp_path / "sample.py"
        original = "x = f(a=1)\n"
        target.write_text(original, encoding = "utf-8")
        monkeypatch.setattr("run_ruff_format.installed_ruff_version", lambda *a, **k: "9.9.9")
        assert main(["--check", str(target)]) == 2
        assert target.read_text(encoding = "utf-8") == original

    def test_check_says_what_the_script_actually_does(self):
        # "unsupported option" alone invites a retry without --check, which would write the file.
        _, error = parse_files(["--check", "any.py"])
        assert error is not None
        assert "--check" in error
        assert "always rewrites" in error

    def test_it_does_not_offer_ruff_as_a_read_only_equivalent(self):
        # `ruff format --check` covers only the middle pass, so it is not a valid substitute.
        _, error = parse_files(["--check", "any.py"])
        assert error is not None
        lowered = error.lower()
        if "ruff" in lowered:
            assert "not an equivalent" in lowered
        assert "diff" in lowered

    @pytest.mark.parametrize("flag", ["-q", "--diff", "--fix", "--unknown"])
    def test_options_are_refused_by_shape_not_by_a_list(self, flag):
        _, error = parse_files([flag])
        assert error is not None and flag in error

    def test_a_missing_path_is_named(self, tmp_path):
        missing = tmp_path / "typo.py"
        files, error = parse_files([str(missing)])
        assert files == []
        assert error is not None and str(missing) in error

    def test_a_missing_path_fails_even_beside_a_real_one(self, tmp_path, monkeypatch):
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
    """Keeps the shebang and executable bit, since people still run the script directly."""

    @pytest.mark.skipif(sys.platform.startswith("win"), reason = "no POSIX mode bits")
    def test_the_formatter_is_executable(self):
        script = _ROOT / "scripts" / "run_ruff_format.py"
        assert script.read_text(encoding = "utf-8").startswith("#!")
        assert (
            script.stat().st_mode & 0o111
        ), "scripts/run_ruff_format.py lost its executable bit; git tracks it as 100755"
