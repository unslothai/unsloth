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

import importlib.util
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


def _probe_result(stdout = "2.9.1+cu128||12.8\n", returncode = 0):
    return MagicMock(returncode = returncode, stdout = stdout)


class TestProbeParsing:
    def test_cuda_build_fields(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result()):
            ran, importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (ran, importable) == (True, True)
        assert (version, hip, cuda) == ("2.9.1+cu128", "", "12.8")

    def test_rocm_build_fields(self):
        out = _probe_result("2.10.0+rocm7.1|7.1.12345|\n")
        with patch.object(stack_mod.subprocess, "run", return_value = out):
            _ran, _importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (version, hip, cuda) == ("2.10.0+rocm7.1", "7.1.12345", "")

    def test_cpu_build_fields(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result("2.9.1||\n")):
            _ran, _importable, version, hip, cuda = stack_mod._probe_torch_runtime()
        assert (version, hip, cuda) == ("2.9.1", "", "")

    def test_last_line_wins_over_import_chatter(self):
        # sitecustomize / import hooks can print before the marker line.
        out = _probe_result("some import warning\n2.9.1+cu128||12.8\n")
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
        assert (ran, importable, version, hip, cuda) == (False, False, "", "", "")

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
            r"import sys; "
            r"sys.stdout.buffer.write(b'noise \xff\xfe\n2.9.1+cu128||12.8\n')"
        )
        real_run = subprocess.run  # bound before the patch, or the stand-in calls itself

        def _emit(_cmd, **kwargs):
            return real_run([sys.executable, "-c", emit], **kwargs)

        with patch.object(
            stack_mod, "_windows_hidden_subprocess_kwargs", lambda: {}
        ), patch.object(stack_mod.subprocess, "run", _emit):
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

        # A repair path reinstalls torch; the next classification must see the new build.
        with (
            patch.object(stack_mod, "USE_UV", False),
            patch.object(stack_mod, "CONSTRAINTS", Path("/nonexistent/constraints.txt")),
            patch.object(
                stack_mod.subprocess, "run", return_value = MagicMock(returncode = 0, stdout = b"")
            ),
        ):
            stack_mod.pip_install("torch repair", "torch")

        out = _probe_result("2.10.0+rocm7.1|7.1.12345|\n")
        with patch.object(stack_mod.subprocess, "run", return_value = out) as mock_run:
            second = stack_mod._probe_torch_runtime()
        assert mock_run.call_count == 1
        assert second[2] == "2.10.0+rocm7.1"

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
            # Second consumer, same run: no new interpreter.
            assert stack_mod._probe_installed_torch_version() == "2.9.1+cu128"
        assert mock_run.call_count == 1

    def test_probe_installed_torch_version_is_none_when_unimportable(self):
        with patch.object(stack_mod.subprocess, "run", return_value = _probe_result("", 1)):
            assert stack_mod._probe_installed_torch_version() is None

    def test_probe_installed_torch_version_is_none_when_stalled(self):
        boom = subprocess.TimeoutExpired(cmd = "python", timeout = 90)
        with patch.object(stack_mod.subprocess, "run", side_effect = boom):
            assert stack_mod._probe_installed_torch_version() is None
