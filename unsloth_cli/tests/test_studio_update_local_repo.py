# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio update --local` must point at a checkout, not at site-packages.

The repo root was derived from __file__, which only holds while the CLI runs from
a source tree. On Windows the first `update --local` replaces the editable
install with a normal one, so the second run derived site-packages and uv failed:

    ERROR: file:///C:/Users/.../unsloth_studio/Lib/site-packages does not appear
    to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.
    [FAILED] Python dependency installation failed (exit code 1)
"""

from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _neutered(monkeypatch):
    """Stub everything update does after resolving the repo root."""
    studio = _studio()
    seen = {}
    monkeypatch.setattr(studio, "_ensure_studio_env_exported", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_release_self_exe_lock_windows", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_cleanup_self_exe_lock_windows", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_refresh_desktop_shortcuts", lambda *a, **k: None)
    monkeypatch.setattr(studio, "_fail_if_install_damaged", lambda *a, **k: None, raising = False)

    def _setup(*a, **k):
        import os
        seen["STUDIO_LOCAL_REPO"] = os.environ.get("STUDIO_LOCAL_REPO")
        seen["STUDIO_LOCAL_INSTALL"] = os.environ.get("STUDIO_LOCAL_INSTALL")

    monkeypatch.setattr(studio, "_run_setup_script", _setup)
    return studio, seen


def test_a_real_checkout_is_passed_through(monkeypatch, tmp_path):
    checkout = tmp_path / "unsloth"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'unsloth'\n")
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(checkout))
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 0, result.output
    assert seen["STUDIO_LOCAL_REPO"] == str(checkout)
    assert seen["STUDIO_LOCAL_INSTALL"] == "1"


def test_site_packages_is_refused_with_an_actionable_message(monkeypatch, tmp_path):
    # What the second `update --local` on Windows actually derived.
    site = tmp_path / "Lib" / "site-packages"
    site.mkdir(parents = True)
    studio, _ = _neutered(monkeypatch)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(site))
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 2, result.output
    out = result.output
    assert "needs an Unsloth checkout" in out
    assert "no pyproject.toml under" in out
    # Both ways forward, because neither is obvious from the uv error it replaces.
    assert "STUDIO_LOCAL_REPO=" in out
    assert "unsloth studio update" in out


def test_the_derived_root_is_used_when_nothing_is_set(monkeypatch):
    # The normal developer case: running from a checkout with no override.
    studio, seen = _neutered(monkeypatch)
    monkeypatch.delenv("STUDIO_LOCAL_REPO", raising = False)
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 0, result.output
    assert Path(seen["STUDIO_LOCAL_REPO"]) == _REPO_ROOT


def test_a_pypi_update_never_looks_for_a_checkout(monkeypatch, tmp_path):
    # Without --local there is no local repo to find, and a stale
    # STUDIO_LOCAL_REPO must not leak into the setup environment.
    site = tmp_path / "site-packages"
    site.mkdir()
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(site))
    result = CliRunner().invoke(studio.studio_app, ["update"])
    assert result.exit_code == 0, result.output
    assert seen["STUDIO_LOCAL_INSTALL"] == "0"
    assert seen["STUDIO_LOCAL_REPO"] is None


def test_a_relative_override_is_absolutised(monkeypatch, tmp_path):
    # setup.sh does `cd "$SCRIPT_DIR"` before install_python_stack.py runs, so
    # a relative path handed straight through resolves against studio/ (which
    # has no pyproject.toml) and hits the exact uv error the guard replaces.
    checkout = tmp_path / "unsloth"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'unsloth'\n")
    studio, seen = _neutered(monkeypatch)
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", ".")
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 0, result.output
    assert Path(seen["STUDIO_LOCAL_REPO"]).is_absolute(), seen["STUDIO_LOCAL_REPO"]
    assert Path(seen["STUDIO_LOCAL_REPO"]) == checkout.resolve()


def test_a_tilde_override_is_expanded(monkeypatch, tmp_path):
    home = tmp_path / "home"
    checkout = home / "unsloth"
    checkout.mkdir(parents = True)
    (checkout / "pyproject.toml").write_text("[project]\nname = 'unsloth'\n")
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("STUDIO_LOCAL_REPO", "~/unsloth")
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 0, result.output
    assert Path(seen["STUDIO_LOCAL_REPO"]) == checkout.resolve()


def test_a_blank_override_falls_back_to_the_derived_root(monkeypatch):
    # `STUDIO_LOCAL_REPO= ` (install.sh resets it to empty) must not become
    # Path(" ") and fail the guard on a perfectly good checkout.
    studio, seen = _neutered(monkeypatch)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", "   ")
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 0, result.output
    assert Path(seen["STUDIO_LOCAL_REPO"]) == _REPO_ROOT


def test_the_override_runs_that_checkouts_setup_script(monkeypatch, tmp_path):
    """The --local checkout's own setup script must win.

    setup.sh/setup.ps1 build the frontend under their own $SCRIPT_DIR, and the
    editable install of the checkout removes the installed tree the installed
    copy's script would have built into. studio/frontend/dist is gitignored, so
    running the installed script against a fresh checkout leaves Studio with no
    frontend at all.
    """
    import platform as _platform

    checkout = tmp_path / "unsloth"
    (checkout / "studio").mkdir(parents = True)
    (checkout / "pyproject.toml").write_text("[project]\nname = 'unsloth'\n")
    name = "setup.ps1" if _platform.system() == "Windows" else "setup.sh"
    script = checkout / "studio" / name
    script.write_text("#!/bin/sh\n")

    studio = _studio()
    assert studio._find_setup_script(checkout) == script
    # No override: unchanged, still resolved from the installed package root.
    assert studio._find_setup_script(None) != script


def test_the_override_reaches_the_setup_runner(monkeypatch, tmp_path):
    checkout = tmp_path / "unsloth"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'unsloth'\n")
    studio, seen = _neutered(monkeypatch)

    def _setup(*a, **k):
        seen["repo_root"] = k.get("repo_root")

    monkeypatch.setattr(studio, "_run_setup_script", _setup)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(checkout))
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 0, result.output
    assert seen["repo_root"] == checkout.resolve()


def test_a_pypi_update_passes_no_checkout(monkeypatch):
    studio, seen = _neutered(monkeypatch)

    def _setup(*a, **k):
        seen["repo_root"] = k.get("repo_root")

    monkeypatch.setattr(studio, "_run_setup_script", _setup)
    result = CliRunner().invoke(studio.studio_app, ["update"])
    assert result.exit_code == 0, result.output
    assert seen["repo_root"] is None


def test_windows_is_shown_a_powershell_assignment(monkeypatch, tmp_path):
    # `VAR=value command` is POSIX shell syntax. PowerShell parses the
    # assignment as a command name, so the only recovery instruction the guard
    # prints was unusable on the platform the guard exists for.
    import platform as _platform

    site = tmp_path / "Lib" / "site-packages"
    site.mkdir(parents = True)
    studio, _ = _neutered(monkeypatch)
    monkeypatch.setattr(_platform, "system", lambda: "Windows")
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(site))
    result = CliRunner().invoke(studio.studio_app, ["update", "--local"])
    assert result.exit_code == 2, result.output
    out = result.output
    assert "$env:STUDIO_LOCAL_REPO=" in out
    # The POSIX prefix form must not be the one Windows is told to run.
    assert "    STUDIO_LOCAL_REPO=/path/to/unsloth" not in out
