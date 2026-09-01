# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The torch classification probe runs once per install run, not once per repair path.

_ensure_cuda_torch / _ensure_xpu_torch / _ensure_rocm_torch / _ensure_cpu_torch all need
the same few facts about the installed torch, and the installer calls the four of them
back to back at two separate repair points. Each used to spawn its own `import torch`,
so a single update paid for up to nine interpreter starts and, on a stalled GPU driver,
up to nine independent 90s timeouts. These tests pin the shared-probe contract: one
subprocess per run, invalidated whenever pip changes what is installed.
"""

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location("studio_install_python_stack", _STACK_PATH)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)


@pytest.fixture(autouse = True)
def _reset_torch_runtime_probe():
    stack_mod._invalidate_torch_runtime_probe()
    yield
    stack_mod._invalidate_torch_runtime_probe()


_MARK = stack_mod._TORCH_PROBE_MARKER


def _probe_result(
    fields = "2.9.1+cu128||12.8",
    returncode = 0,
    raw = None,
):
    """A probe stdout carrying our marked line, plus whatever chatter is asked for."""
    return MagicMock(
        returncode = returncode,
        stdout = raw if raw is not None else (f"{_MARK}{fields}\n" if fields else ""),
    )


class TestProbeParsing:
    def test_cuda_build_fields(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result()):
            ran, importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (ran, importable) == (True, True)
        assert (version, hip, cuda) == ("2.9.1+cu128", "", "12.8")

    def test_rocm_build_fields(self):
        out = _probe_result("2.10.0+rocm7.1|7.1.12345|")
        with patch.object(stack_mod.subprocess, "run", return_value = out):
            _ran, _importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (version, hip, cuda) == ("2.10.0+rocm7.1", "7.1.12345", "")

    def test_cpu_build_fields(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result("2.9.1||")):
            _ran, _importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (version, hip, cuda) == ("2.9.1", "", "")

    def test_last_line_wins_over_import_chatter(self):
        # sitecustomize / import hooks can print before the marker line.
        out = _probe_result(raw = f"some import warning\n{_MARK}2.9.1+cu128||12.8\n")
        with patch.object(stack_mod.subprocess, "run", return_value = out):
            _ran, _importable, version, _hip, cuda = stack_mod._probe_torch_runtime()
        assert (version, cuda) == ("2.9.1+cu128", "12.8")

    def test_unimportable_torch_is_distinguished_from_a_stalled_probe(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result("", 1)):
            ran, importable, _version, _hip, _cuda = stack_mod._probe_torch_runtime()
        # ran=True lets callers force a repair; a stalled probe (ran=False) must not.
        assert (ran, importable) == (True, False)

    def test_timeout_reports_not_ran(self):
        boom = subprocess.TimeoutExpired(cmd = "python", timeout = 90)
        with patch.object(stack_mod.subprocess, "run", side_effect = boom):
            ran, importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (ran, importable, version, hip, cuda) == (False, False, None, "", "")

    def test_chatter_after_the_marker_does_not_win(self):
        """An atexit handler, a CUDA teardown notice or a "Segmentation fault" line can
        arrive AFTER the answer, so "the last non-empty line" is not reliably ours."""
        out = _probe_result(raw = f"{_MARK}2.9.1+cu128||12.8\ndestroying CUDA context\n")
        with patch.object(stack_mod.subprocess, "run", return_value = out):
            _ran, _importable, version, _hip, cuda = stack_mod._probe_torch_runtime()
        assert (version, cuda) == ("2.9.1+cu128", "12.8")

    def test_no_marked_line_reports_an_unknown_version(self):
        """Exit 0 with nothing of ours on stdout means we learned nothing. None, not "",
        because the XPU and CPU pins act on an empty version and must not act on this."""
        out = _probe_result(raw = "only chatter, no answer\n")
        with patch.object(stack_mod.subprocess, "run", return_value = out):
            ran, importable, version, _hip, _cuda = stack_mod._probe_torch_runtime()
        assert (ran, importable, version) == (True, True, None)

    def test_an_empty_version_is_reported_as_empty_not_unknown(self):
        """A torch whose __version__ is empty IS broken, and an XPU pin repairs it.
        Collapsing that into the unknown case silently skips the repair."""
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result("||")):
            _ran, _importable, version, _hip, _cuda = stack_mod._probe_torch_runtime()
        assert version == ""

    def test_a_torch_without_a_version_module_still_classifies(self, tmp_path):
        """torch.version is not guaranteed to exist. Reaching through it unguarded raises
        inside the child, which reads as "torch cannot import" and force-reinstalls a
        working venv. Runs the real subprocess against a real package on PYTHONPATH,
        since a mock cannot show which attribute the child touched."""
        pkg = tmp_path / "torch"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("__version__ = '1.13.1'\n", encoding = "utf-8")
        with patch.dict(os.environ, {"PYTHONPATH": str(tmp_path)}):
            ran, importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (ran, importable, version) == (True, True, "1.13.1")
        assert (hip, cuda) == ("", "")

    def test_oserror_reports_not_ran(self):
        with patch.object(stack_mod.subprocess, "run", side_effect = OSError("no exe")):
            ran, _importable, _version, _hip, _cuda = stack_mod._probe_torch_runtime()
        assert ran is False

    def test_undecodable_import_chatter_does_not_escape(self):
        """The probes this replaced all decoded with errors="replace".

        text=True on its own decodes strictly, and UnicodeDecodeError is a ValueError, so
        one undecodable byte from torch's import chatter would sail past the except above
        and take the whole installer down rather than falling back to the on-disk
        classifier. Runs the real subprocess: a mock cannot show which decoder was used.
        """
        emit = (
            "import sys; sys.stdout.buffer.write("
            r"b'noise \xff\xfe\n' + " + repr(_MARK) + r".encode() + b'2.9.1+cu128||12.8\n')"
        )
        real_run = subprocess.run  # bound before the patch, or the stand-in calls itself

        def _emit(_cmd, **kwargs):
            return real_run([sys.executable, "-c", emit], **kwargs)

        with (
            patch.object(stack_mod, "_windows_hidden_subprocess_kwargs", lambda: {}),
            patch.object(stack_mod.subprocess, "run", _emit),
        ):
            ran, importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (ran, importable) == (True, True)
        assert (version, hip, cuda) == ("2.9.1+cu128", "", "12.8")


class TestMemoization:
    def test_repeated_calls_spawn_one_interpreter(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result()) as mock_run:
            for _ in range(5):
                stack_mod._probe_torch_runtime()
        assert mock_run.call_count == 1

    def test_a_stalled_probe_is_not_retried(self):
        # The whole point: nine 90s waits become one.
        boom = subprocess.TimeoutExpired(cmd = "python", timeout = 90)
        with patch.object(stack_mod.subprocess, "run", side_effect = boom) as mock_run:
            for _ in range(5):
                stack_mod._probe_torch_runtime()
        assert mock_run.call_count == 1

    def test_pip_install_invalidates_the_cache(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result()):
            first = stack_mod._probe_torch_runtime()
        assert first[2] == "2.9.1+cu128"

        # A repair path reinstalls torch;
        with (
            patch.object(stack_mod, "USE_UV", False),
            patch.object(stack_mod, "CONSTRAINTS", Path("/nonexistent/constraints.txt")),
            patch.object(
                stack_mod.subprocess, "run", return_value = MagicMock(returncode = 0, stdout = b"")
            ),
        ):
            stack_mod.pip_install("torch repair", "torch")

        out = _probe_result("2.10.0+rocm7.1|7.1.12345|")
        with patch.object(stack_mod.subprocess, "run", return_value = out) as mock_run:
            second = stack_mod._probe_torch_runtime()
        assert mock_run.call_count == 1
        assert second[2] == "2.10.0+rocm7.1"

    def test_pip_install_try_invalidates_the_cache(self):
        """The other installer. It puts the Windows AMD ROCm trio on disk, so a memo it
        does not clear can answer for the build it just replaced."""
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result()):
            assert stack_mod._probe_torch_runtime()[2] == "2.9.1+cu128"

        with (
            patch.object(stack_mod, "USE_UV", False),
            patch.object(stack_mod, "CONSTRAINTS", Path("/nonexistent/constraints.txt")),
            patch.object(
                stack_mod.subprocess, "run", return_value = MagicMock(returncode = 0, stdout = b"")
            ),
        ):
            assert stack_mod.pip_install_try("ROCm torch (Windows)", "torch") is True

        out = _probe_result("2.10.0+rocm7.1|7.1.12345|")
        with patch.object(stack_mod.subprocess, "run", return_value = out) as mock_run:
            assert stack_mod._probe_torch_runtime()[2] == "2.10.0+rocm7.1"
        assert mock_run.call_count == 1

    def test_the_torchao_probe_sees_the_reinstalled_torch(self):
        """The consumer that would actually read a stale answer. _select_torchao_spec
        reads _probe_installed_torch_version() between the two repair points, so a memo
        surviving the reinstall pins torchao against the torch that was just replaced.
        """
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result()):
            assert stack_mod._probe_installed_torch_version() == "2.9.1+cu128"

        with (
            patch.object(stack_mod, "USE_UV", False),
            patch.object(stack_mod, "CONSTRAINTS", Path("/nonexistent/constraints.txt")),
            patch.object(
                stack_mod.subprocess, "run", return_value = MagicMock(returncode = 0, stdout = b"")
            ),
        ):
            assert stack_mod.pip_install_try("ROCm torch (Windows)", "torch") is True

        out = _probe_result("2.10.0+rocm7.1|7.1.12345|")
        with patch.object(stack_mod.subprocess, "run", return_value = out):
            assert stack_mod._probe_installed_torch_version() == "2.10.0+rocm7.1"

    @pytest.mark.parametrize("installer", ["pip_install", "pip_install_try"])
    def test_a_real_reinstall_is_really_reclassified(self, tmp_path, installer):
        """The mocked versions above prove the memo was dropped. This proves the answer
        that replaces it comes from the venv as it is NOW: two real torch packages, a
        real probe subprocess either side of a real call into the installer.
        """

        def _torch(where, version):
            pkg = where / "torch"
            pkg.mkdir(parents = True)
            (pkg / "__init__.py").write_text(
                "from . import version\nfrom .version import __version__\n", encoding = "utf-8"
            )
            (pkg / "version.py").write_text(
                f"__version__ = '{version}'\nhip = None\ncuda = None\n", encoding = "utf-8"
            )
            return where

        before = _torch(tmp_path / "before", "2.9.1+cpu")
        after = _torch(tmp_path / "after", "2.10.0+cu128")

        with patch.dict(os.environ, {"PYTHONPATH": str(before)}):
            assert stack_mod._probe_torch_runtime()[2] == "2.9.1+cpu"
        # Still the remembered answer while nothing has installed anything.
        with patch.dict(os.environ, {"PYTHONPATH": str(after)}):
            assert stack_mod._probe_torch_runtime()[2] == "2.9.1+cpu"

        with (
            patch.object(stack_mod, "USE_UV", False),
            patch.object(stack_mod, "CONSTRAINTS", Path("/nonexistent/constraints.txt")),
            patch.object(
                stack_mod.subprocess, "run", return_value = MagicMock(returncode = 0, stdout = b"")
            ),
        ):
            getattr(stack_mod, installer)("torch repair", "torch")

        with patch.dict(os.environ, {"PYTHONPATH": str(after)}):
            assert stack_mod._probe_torch_runtime()[2] == "2.10.0+cu128"

    def test_every_installer_entry_point_invalidates(self):
        """Read from the module rather than listed here, so a third installer helper
        cannot be added without either invalidating or failing this.

        The two that exist route through _build_pip_cmd / _build_uv_cmd, which is what
        makes a function an installer rather than a probe.
        """
        tree = ast.parse(Path(stack_mod.__file__).read_text(encoding = "utf-8"))
        installers = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            called = {
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            if called & {"_build_pip_cmd", "_build_uv_cmd"}:
                installers[node.name] = called
        assert set(installers) == {"pip_install", "pip_install_try"}, (
            f"a new installer entry point appeared: {sorted(installers)}. It has to drop "
            "the torch classification too, or it will answer for the build it replaced"
        )
        for name, called in installers.items():
            assert "_invalidate_torch_runtime_probe" in called, (
                f"{name}() installs packages without dropping the memoized torch " "classification"
            )

    def test_explicit_invalidation_forces_a_reprobe(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result()) as mock_run:
            stack_mod._probe_torch_runtime()
            stack_mod._invalidate_torch_runtime_probe()
            stack_mod._probe_torch_runtime()
        assert mock_run.call_count == 2


class TestConsumersShareTheProbe:
    def test_probe_installed_torch_version_uses_the_shared_result(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result()) as mock_run:
            assert stack_mod._probe_installed_torch_version() == "2.9.1+cu128"
            assert stack_mod._probe_installed_torch_version() == "2.9.1+cu128"
        assert mock_run.call_count == 1

    def test_probe_installed_torch_version_is_none_when_unimportable(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result("", 1)):
            assert stack_mod._probe_installed_torch_version() is None

    def test_probe_installed_torch_version_is_none_when_stalled(self):
        boom = subprocess.TimeoutExpired(cmd = "python", timeout = 90)
        with patch.object(stack_mod.subprocess, "run", side_effect = boom):
            assert stack_mod._probe_installed_torch_version() is None


class TestVersionlessBuildsStillClassify:
    """An empty version field is not no answer: "" is a torch whose __version__ is
    missing, which the pins repair, and None is a probe that learned nothing and must
    leave the venv alone. TestProbeParsing pins that at the probe; these pin it where
    it decides something, since gating on the version alone would skip the repair.
    """

    @patch.object(stack_mod, "NO_TORCH", False)
    @patch.object(stack_mod, "pip_install")
    def test_cpu_pin_still_replaces_a_versionless_cuda_build(self, mock_pip):
        out = _probe_result("||12.8")
        with patch.object(
            stack_mod,
            "_explicit_cpu_torch_index_url",
            return_value = "https://download.pytorch.org/whl/cpu",
        ):
            with patch.object(stack_mod.subprocess, "run", return_value = out):
                stack_mod._ensure_cpu_torch()
        assert mock_pip.called, "a CUDA build under an explicit CPU pin must be replaced"

    @patch.object(stack_mod, "NO_TORCH", False)
    @patch.object(stack_mod, "pip_install")
    def test_cpu_pin_still_replaces_a_versionless_rocm_build(self, mock_pip):
        out = _probe_result("|7.1.12345|")
        with patch.object(
            stack_mod,
            "_explicit_cpu_torch_index_url",
            return_value = "https://download.pytorch.org/whl/cpu",
        ):
            with patch.object(stack_mod.subprocess, "run", return_value = out):
                stack_mod._ensure_cpu_torch()
        assert mock_pip.called, "a ROCm build under an explicit CPU pin must be replaced"

    @patch.object(stack_mod, "NO_TORCH", False)
    @patch.object(stack_mod, "pip_install")
    def test_cpu_pin_leaves_a_cpu_build_with_no_version_alone(self, mock_pip):
        # Every field empty is a CPU build with an unreadable version:
        with patch.object(
            stack_mod,
            "_explicit_cpu_torch_index_url",
            return_value = "https://download.pytorch.org/whl/cpu",
        ):
            with patch.object(stack_mod.subprocess, "run", return_value = _probe_result("||")):
                stack_mod._ensure_cpu_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "NO_TORCH", False)
    @patch.object(stack_mod, "pip_install")
    def test_cpu_pin_leaves_the_venv_alone_when_the_probe_said_nothing(self, mock_pip):
        # Exit 0 with no line of ours:
        with patch.object(
            stack_mod,
            "_explicit_cpu_torch_index_url",
            return_value = "https://download.pytorch.org/whl/cpu",
        ):
            with patch.object(
                stack_mod.subprocess, "run", return_value = _probe_result(raw = "unrelated chatter\n")
            ):
                stack_mod._ensure_cpu_torch()
        mock_pip.assert_not_called()

    @patch.object(stack_mod, "NO_TORCH", False)
    @patch.object(stack_mod, "IS_MACOS", False)
    @patch.object(stack_mod, "IS_WINDOWS", False)
    @patch.object(stack_mod, "pip_install")
    def test_xpu_pin_still_repairs_a_versionless_build(self, mock_pip):
        # An unreadable version is not a supported +xpu build, and the pin forces the family.
        with patch.object(
            stack_mod,
            "_explicit_xpu_torch_index_url",
            return_value = "https://download.pytorch.org/whl/xpu",
        ):
            with patch.object(stack_mod.subprocess, "run", return_value = _probe_result("||")):
                stack_mod._ensure_xpu_torch()
        assert mock_pip.called, "an unidentifiable build under an explicit XPU pin must be repaired"
