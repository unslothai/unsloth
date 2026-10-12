# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An update must finish with the installer it installs, not the one it started with.

The core-packages step upgrades the package that ships install_python_stack.py. Before this, the
old process completed its own step list, so every step a release added (the pinned Diffusers main
build in 2026.9.8) was skipped by the update that installed it.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

STACK = Path(__file__).resolve().parents[3] / "studio" / "install_python_stack.py"


def _module(name = "install_python_stack_rerun_probe"):
    spec = importlib.util.spec_from_file_location(name, STACK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_unchanged_installer_is_not_rerun(monkeypatch):
    module = _module()
    monkeypatch.delenv(module._INSTALLER_RERUN_ENV, raising = False)
    assert module._installer_replaced() is False


def test_a_replaced_installer_is_rerun_once(monkeypatch):
    module = _module()
    monkeypatch.delenv(module._INSTALLER_RERUN_ENV, raising = False)
    monkeypatch.setattr(module, "_read_own_source", lambda: b"the next release")
    assert module._installer_replaced() is True
    # The rerun carries the marker, so a second replacement cannot loop.
    monkeypatch.setenv(module._INSTALLER_RERUN_ENV, "1")
    assert module._installer_replaced() is False


def test_an_unreadable_installer_keeps_the_running_one(monkeypatch):
    module = _module()
    monkeypatch.delenv(module._INSTALLER_RERUN_ENV, raising = False)
    monkeypatch.setattr(module, "_read_own_source", lambda: None)
    assert module._installer_replaced() is False


def test_the_rerun_runs_the_file_on_disk_and_returns_its_exit_code(monkeypatch):
    module = _module()
    calls = []

    def fake_run(argv, env):
        calls.append((argv, env))
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [str(STACK)])
    assert module._rerun_replaced_installer() == 7
    argv, env = calls[0]
    assert argv == [sys.executable, str(STACK.resolve())]
    assert env[module._INSTALLER_RERUN_ENV] == "1"


def test_the_check_follows_the_core_step_and_the_rerun_starts_outside_the_pass_lock():
    """Raised inside install_python_stack, caught only at the entry point.

    Structural, because driving the real pass needs a venv and the network.
    """
    source = STACK.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "install_python_stack"
    )
    body = ast.get_source_segment(source, func)
    core = body.rindex('"Updating core packages"')
    check = body.index("_installer_replaced()")
    assert core < check < body.index("_mlx_vlm_spec_now = ")
    main = source[source.index('if __name__ == "__main__":') :]
    assert "except _InstallerReplaced:" in main
    assert "_rerun_replaced_installer()" in main
