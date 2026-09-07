# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Terminal refusal is a product outcome, separate from opt-in compatibility diagnostics."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import terminal_qualification as qualification
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


@pytest.mark.parametrize(
    "code", ["WINDOWS_SANDBOX_TERMINAL_PROBE_FAILED", "WINDOWS_SANDBOX_CANCELLED"]
)
def test_failed_terminal_qualification_preserves_exact_refusal_without_retry(
    monkeypatch, tmp_path, code
):
    calls = []
    selected = r"C:\Windows\System32\cmd.exe"
    cancel = object()

    def fail(executable, **kwargs):
        calls.append((executable, kwargs))
        raise WindowsRuntimeError(code, "quoted_script failed with exit status 1")

    monkeypatch.setattr(qualification, "run_terminal_probe", fail)
    result = qualification.qualify_terminal_runtime(selected, store_root = tmp_path, cancel = cancel)
    assert not result.qualified
    assert result.failure_code == code
    assert "quoted_script failed with exit status 1" in result.reason
    assert result.limitations == ("terminal_runtime_unqualified",)
    assert result.transient == (code == "WINDOWS_SANDBOX_CANCELLED")
    assert calls == [(selected, dict(store_root = tmp_path, timeout = 30, cancel = cancel))]
    assert not list(tmp_path.iterdir())


def test_compatibility_success_does_not_grant_terminal_qualification(monkeypatch, tmp_path):
    observed = SimpleNamespace(
        selected_executable = "selected",
        checks = ("fixed_control",),
        runtime_digest = "a" * 64,
        content_digest = "b" * 64,
    )
    monkeypatch.setattr(qualification, "run_terminal_probe", lambda *args, **kwargs: observed)
    result = qualification.qualify_terminal_runtime("selected", store_root = tmp_path)
    assert not result.qualified
    assert result.failure_code == "WINDOWS_SANDBOX_QUALIFICATION_INCOMPLETE"
    assert result.runtime_digest == observed.runtime_digest
    assert result.content_digest == observed.content_digest
    assert result.limitations == ("terminal_enforcement_unqualified",)


@pytest.mark.skipif(sys.platform != "win32", reason = "Native Terminal refusal")
def test_real_selected_cmd_remains_unqualified_with_explicit_reason(tmp_path):
    selected = str(Path(os.environ["SystemRoot"]) / "System32/cmd.exe")
    result = qualification.qualify_terminal_runtime(selected, store_root = str(tmp_path / "store"))
    assert not result.qualified
    assert result.failure_code in {
        "WINDOWS_SANDBOX_TERMINAL_PROBE_FAILED",
        "WINDOWS_SANDBOX_QUALIFICATION_INCOMPLETE",
    }, result.reason
    if result.failure_code == "WINDOWS_SANDBOX_TERMINAL_PROBE_FAILED":
        assert "quoted_script" in result.reason and "exited with 1" in result.reason
    assert result.selected_executable == selected
    assert result.limitations
