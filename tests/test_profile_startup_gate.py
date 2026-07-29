# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression coverage for the startup profiler's budget gate and process teardown."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "profile_startup.py"


def _load():
    spec = importlib.util.spec_from_file_location("profile_startup", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_budget_fails_when_no_launch_was_measured(capsys, monkeypatch):
    """A requested budget must not pass just because the CLI was never found."""
    mod = _load()
    monkeypatch.setattr(mod, "find_bin", lambda: None)
    monkeypatch.setattr(mod, "profile_imports", lambda python, top = 15: {"ok": False, "error": ""})
    rc = mod.main(["--max-healthz-seconds", "30"])
    assert rc == 1
    assert "no healthz measurement" in capsys.readouterr().out


def test_budget_rejects_import_only():
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main(["--import-only", "--max-healthz-seconds", "30"])
    assert exc.value.code == 2


def test_terminate_tree_falls_back_when_taskkill_fails(monkeypatch):
    """A nonzero taskkill must still reach terminate(), not return silently."""
    mod = _load()
    monkeypatch.setattr(mod.os, "name", "nt")
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 1, "", "ERROR"),
    )

    class _Proc:
        pid = 4321
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            type(self).terminated = True

    mod._terminate_tree(_Proc())
    assert _Proc.terminated


def test_terminate_tree_returns_on_successful_taskkill(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.os, "name", "nt")
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, "", ""),
    )

    class _Proc:
        pid = 4321
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            type(self).terminated = True

    mod._terminate_tree(_Proc())
    assert not _Proc.terminated


@pytest.mark.skipif(sys.platform == "win32", reason = "posix branch")
def test_terminate_tree_posix_uses_terminate(monkeypatch):
    mod = _load()

    class _Proc:
        pid = 1
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            type(self).terminated = True

    mod._terminate_tree(_Proc())
    assert _Proc.terminated
