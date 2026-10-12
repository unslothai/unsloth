# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio update` must tell setup.ps1 which interpreter launched it.

The CLI runs from the managed venv's own python.exe, and setup.ps1's stale-venv branch
used to Remove-Item that venv: Windows will not delete a running image, so the delete
emptied Lib\\ and failed on Scripts\\python.exe, leaving an environment with no unsloth_cli
and no rollback copy. setup.ps1 now repairs in place when it sees it runs from inside the
venv, and UNSLOTH_SETUP_HOST_PYTHON is how it sees that without a process walk.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _setup_tree(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "studio").mkdir(parents = True, exist_ok = True)
    (repo_root / "studio" / "setup.sh").write_text("")
    (repo_root / "studio" / "setup.ps1").write_text("")
    return repo_root


class _Result:
    returncode = 0


def _quiet_cache(monkeypatch, studio) -> None:
    # The uv cache seeding has its own tests; here it must neither spawn uv nor read the machine.
    monkeypatch.setattr(studio, "_with_studio_uv_cache", lambda env, cwd = None: env)
    monkeypatch.setattr(studio, "_backfill_uv_cache_marker", lambda env: None)


def test_the_posix_update_names_its_own_interpreter(monkeypatch, tmp_path):
    studio = _studio()
    _quiet_cache(monkeypatch, studio)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    seen: dict = {}

    def _fake_run(
        argv,
        env = None,
        **kwargs,
    ):
        seen["env"] = env
        return _Result()

    monkeypatch.setattr(studio.subprocess, "run", _fake_run)
    studio._run_setup_script(repo_root = _setup_tree(tmp_path))

    assert seen["env"] is not None, "env must be materialised to carry the interpreter"
    assert seen["env"]["UNSLOTH_SETUP_HOST_PYTHON"] == sys.executable


def test_the_windows_update_names_its_own_interpreter(monkeypatch, tmp_path):
    """The PowerShell spawn is the one that matters: setup.ps1 is where the venv was deleted."""
    studio = _studio()
    _quiet_cache(monkeypatch, studio)
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        studio._studio_runtime_gate, "resolve_windows_powershell", lambda: "powershell.exe"
    )
    monkeypatch.setattr(studio, "_probe_profile_proxy_defaults", lambda hosts: None)
    monkeypatch.setattr(studio, "_wait_for_windows_setup_process", lambda process: 0)
    seen: dict = {}

    class _Process:
        pass

    def _fake_popen(
        argv,
        env = None,
        **kwargs,
    ):
        seen["env"] = env
        return _Process()

    monkeypatch.setattr(studio.subprocess, "Popen", _fake_popen)
    studio._run_setup_script(repo_root = _setup_tree(tmp_path))

    assert seen["env"]["UNSLOTH_SETUP_HOST_PYTHON"] == sys.executable


def test_the_verbose_flag_survives_alongside_it(monkeypatch, tmp_path):
    """The verbose branch builds env first; the interpreter must extend it, not replace it."""
    studio = _studio()
    _quiet_cache(monkeypatch, studio)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    seen: dict = {}

    def _fake_run(
        argv,
        env = None,
        **kwargs,
    ):
        seen["env"] = env
        return _Result()

    monkeypatch.setattr(studio.subprocess, "run", _fake_run)
    studio._run_setup_script(verbose = True, repo_root = _setup_tree(tmp_path))

    assert seen["env"]["UNSLOTH_VERBOSE"] == "1"
    assert seen["env"]["UNSLOTH_SETUP_HOST_PYTHON"] == sys.executable
