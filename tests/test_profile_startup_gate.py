# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression coverage for the startup profiler's budget gate and process teardown."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "profile_startup.py"


def _load():
    spec = importlib.util.spec_from_file_location("profile_startup", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _no_subprocesses(mod, monkeypatch):
    # Keep the gate tests off the real interpreter and CLI.
    monkeypatch.setattr(mod, "find_bin", lambda: None)
    monkeypatch.setattr(mod, "profile_imports", lambda python, top = 15: {"ok": False, "error": ""})
    monkeypatch.setattr(mod, "python_version_of", lambda python: "3.13.0")


class _Proc:
    """Stand-in for a still-running Popen."""

    def __init__(self):
        self.pid = 4321
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


def _nt(mod, monkeypatch, returncode):
    calls: list[list[str]] = []

    def _run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, "", "")

    # Patch the module's own references, not the real os/subprocess the session shares.
    monkeypatch.setattr(mod, "os", SimpleNamespace(name = "nt"))
    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run = _run))
    return calls


def test_budget_fails_when_no_launch_was_measured(capsys, monkeypatch):
    """A requested budget must not pass just because the CLI was never found."""
    mod = _load()
    _no_subprocesses(mod, monkeypatch)
    rc = mod.main(["--max-healthz-seconds", "30"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "::error::" in out and "no healthz measurement" in out
    assert "no unsloth CLI found" in out


def test_budget_still_passes_when_a_launch_was_measured(monkeypatch):
    """The fail-closed branch must not swallow a genuinely healthy run."""
    mod = _load()
    _no_subprocesses(mod, monkeypatch)
    monkeypatch.setattr(mod, "find_bin", lambda: "unsloth")
    monkeypatch.setattr(
        mod,
        "profile_launch",
        lambda bin_path, port, **kw: {
            "spawn_seconds": 0.1,
            "healthz_seconds": 1.5,
            "lifespan_ms": 100.0,
            "reached_healthz": True,
            "log_tail": [],
        },
    )
    assert mod.main(["--max-healthz-seconds", "30"]) == 0
    assert mod.main(["--max-healthz-seconds", "1"]) == 1


def test_budget_rejects_import_only(capsys):
    """--import-only launches nothing, so a budget on it could only ever pass."""
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main(["--import-only", "--max-healthz-seconds", "30"])
    assert exc.value.code == 2
    assert "--import-only" in capsys.readouterr().err


def test_terminate_tree_falls_back_when_taskkill_fails(monkeypatch):
    """A nonzero taskkill must still reach terminate(), not return silently."""
    mod = _load()
    calls = _nt(mod, monkeypatch, returncode = 1)
    proc = _Proc()
    mod._terminate_tree(proc)
    assert calls == [["taskkill", "/PID", "4321", "/T", "/F"]]
    assert proc.terminated


def test_terminate_tree_falls_back_when_taskkill_raises(monkeypatch):
    """A missing or hung taskkill must reach terminate() too."""
    mod = _load()
    monkeypatch.setattr(mod, "os", SimpleNamespace(name = "nt"))

    def _boom(argv, **kwargs):
        raise FileNotFoundError(argv)

    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run = _boom))
    proc = _Proc()
    mod._terminate_tree(proc)
    assert proc.terminated


def test_terminate_tree_returns_on_successful_taskkill(monkeypatch):
    mod = _load()
    _nt(mod, monkeypatch, returncode = 0)
    proc = _Proc()
    mod._terminate_tree(proc)
    assert not proc.terminated


def test_terminate_tree_skips_an_exited_process(monkeypatch):
    mod = _load()
    calls = _nt(mod, monkeypatch, returncode = 0)
    proc = _Proc()
    proc.poll = lambda: 0
    mod._terminate_tree(proc)
    assert calls == [] and not proc.terminated


@pytest.mark.skipif(sys.platform == "win32", reason = "posix branch")
def test_terminate_tree_posix_uses_terminate():
    mod = _load()
    proc = _Proc()
    mod._terminate_tree(proc)
    assert proc.terminated
