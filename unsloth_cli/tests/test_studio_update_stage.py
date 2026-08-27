# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli import _studio_stage  # noqa: E402


def _make_venv(root: Path, prefix: Path | None = None) -> Path:
    venv = root / _studio_stage.VENV_NAME
    prefix = prefix or venv
    (venv / "bin").mkdir(parents = True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion_info = 3.13\n", encoding = "utf-8")
    (venv / "bin" / "unsloth").write_text(
        f"#!{prefix}/bin/python\nimport sys\nprint('cli')\n", encoding = "utf-8"
    )
    (venv / "bin" / "pip").write_text(
        f"#!{prefix}/bin/python3.13\nprint('pip')\n", encoding = "utf-8"
    )
    (venv / "bin" / "activate").write_text(f"VIRTUAL_ENV='{prefix}'\n", encoding = "utf-8")
    (venv / "bin" / "env-script").write_text("#!/usr/bin/env python\nprint(1)\n", encoding = "utf-8")
    (venv / "bin" / "native").write_bytes(b"\x7fELF\x02\x01\x01")
    return venv


def test_runtime_root_follows_the_stage_override(monkeypatch, tmp_path):
    monkeypatch.delenv(_studio_stage.STAGE_ROOT_ENV, raising = False)
    assert _studio_stage.runtime_root(tmp_path) == tmp_path
    assert not _studio_stage.is_staging()

    monkeypatch.setenv(_studio_stage.STAGE_ROOT_ENV, str(tmp_path / "stage"))
    assert _studio_stage.runtime_root(tmp_path) == tmp_path / "stage"
    assert _studio_stage.is_staging()


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX shebangs")
def test_make_relocatable_rewrites_only_venv_python_shebangs(tmp_path):
    venv = _make_venv(tmp_path)

    rewritten = _studio_stage.make_relocatable(venv)

    assert rewritten == 2
    for name in ("unsloth", "pip"):
        text = (venv / "bin" / name).read_text(encoding = "utf-8")
        assert text.startswith(_studio_stage.RELOCATABLE_SHEBANG)
        assert "realpath" in text.splitlines()[1]
    assert (venv / "bin" / "unsloth").read_text(encoding = "utf-8").endswith("print('cli')\n")
    assert (
        (venv / "bin" / "env-script")
        .read_text(encoding = "utf-8")
        .startswith("#!/usr/bin/env python")
    )
    assert (venv / "bin" / "native").read_bytes() == b"\x7fELF\x02\x01\x01"
    assert (venv / "bin" / "activate").read_text(encoding = "utf-8") == f"VIRTUAL_ENV='{venv}'\n"
    assert "relocatable = true" in (venv / "pyvenv.cfg").read_text(encoding = "utf-8")


def test_make_relocatable_does_not_duplicate_the_flag(tmp_path):
    venv = _make_venv(tmp_path)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\nrelocatable = true\n", encoding = "utf-8")

    _studio_stage.make_relocatable(venv)

    assert (venv / "pyvenv.cfg").read_text(encoding = "utf-8").count("relocatable") == 1


def test_clone_tree_copies_symlinks_as_symlinks(tmp_path):
    source = tmp_path / "src"
    (source / "bin").mkdir(parents = True)
    (source / "bin" / "real").write_text("x", encoding = "utf-8")
    os.symlink("real", source / "bin" / "link")

    _studio_stage.clone_tree(source, tmp_path / "dst")

    assert (tmp_path / "dst" / "bin" / "link").is_symlink()
    assert (tmp_path / "dst" / "bin" / "link").read_text(encoding = "utf-8") == "x"


def test_stage_builds_a_ready_marker_from_a_successful_update(monkeypatch, tmp_path):
    home = tmp_path / "studio"
    _make_venv(home)
    monkeypatch.setenv(_studio_stage.SHELL_VERSION_ENV, "0.1.900-beta")
    seen: dict = {}

    def fake_update(root: Path, args: list[str]) -> int:
        seen["root"] = root
        seen["args"] = args
        return 0

    monkeypatch.setattr(_studio_stage, "installed_version", lambda venv, env: "2026.9.1")
    monkeypatch.setattr(_studio_stage, "probe_cli", lambda venv, env: None)
    echoed: list[str] = []

    result = _studio_stage.stage(
        home, update_args = ["--package", "unsloth"], echo = echoed.append, run_update = fake_update
    )

    root = home / _studio_stage.STAGE_DIR_NAME
    assert seen == {"root": root, "args": ["--package", "unsloth"]}
    assert result == {"backend_version": "2026.9.1", "root": str(root)}
    marker = json.loads((root / _studio_stage.READY_MARKER).read_text(encoding = "utf-8"))
    assert marker["backend_version"] == "2026.9.1"
    assert marker["shell_version"] == "0.1.900-beta"
    assert (root / _studio_stage.VENV_NAME / "pyvenv.cfg").is_file()
    assert (
        (home / _studio_stage.VENV_NAME / "bin" / "unsloth")
        .read_text(encoding = "utf-8")
        .startswith(f"#!{home / _studio_stage.VENV_NAME}/bin/python")
    )
    assert echoed == ["[TAURI:STEP] clone", "[TAURI:STEP] update", "[TAURI:STEP] verify"]


def test_stage_discards_the_tree_when_the_update_fails(monkeypatch, tmp_path):
    home = tmp_path / "studio"
    _make_venv(home)
    monkeypatch.delenv(_studio_stage.SHELL_VERSION_ENV, raising = False)

    with pytest.raises(_studio_stage.StageError, match = "staged update failed"):
        _studio_stage.stage(
            home, update_args = [], echo = lambda _: None, run_update = lambda root, args: 1
        )

    assert not (home / _studio_stage.STAGE_DIR_NAME).exists()
    assert (home / _studio_stage.VENV_NAME / "pyvenv.cfg").is_file()


def test_stage_discards_the_tree_when_verification_fails(monkeypatch, tmp_path):
    home = tmp_path / "studio"
    _make_venv(home)

    def broken(venv: Path, env: dict) -> str:
        raise _studio_stage.StageError("no unsloth")

    monkeypatch.setattr(_studio_stage, "installed_version", broken)

    with pytest.raises(_studio_stage.StageError, match = "no unsloth"):
        _studio_stage.stage(
            home, update_args = [], echo = lambda _: None, run_update = lambda root, args: 0
        )

    assert not (home / _studio_stage.STAGE_DIR_NAME).exists()


def test_stage_refuses_without_a_managed_environment(tmp_path):
    with pytest.raises(_studio_stage.StageError, match = "no managed environment"):
        _studio_stage.stage(
            tmp_path, update_args = [], echo = lambda _: None, run_update = lambda root, args: 0
        )


def test_child_environment_points_the_staged_cli_at_the_stage_root(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRTUAL_ENV", "/elsewhere")
    monkeypatch.setenv("PYTHONHOME", "/elsewhere")

    env = _studio_stage.child_environment(tmp_path)

    assert env[_studio_stage.STAGE_ROOT_ENV] == str(tmp_path)
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONHOME" not in env


def test_update_stage_flag_refuses_local(monkeypatch):
    from typer.testing import CliRunner
    from unsloth_cli.commands import studio as studio_mod

    monkeypatch.delenv(_studio_stage.STAGE_ROOT_ENV, raising = False)
    result = CliRunner().invoke(studio_mod.studio_app, ["update", "--stage", "--local"])

    assert result.exit_code == 2
    assert "--stage cannot be combined with --local" in result.output


def test_update_stage_reports_a_structured_error(monkeypatch):
    from typer.testing import CliRunner
    from unsloth_cli.commands import studio as studio_mod

    monkeypatch.delenv(_studio_stage.STAGE_ROOT_ENV, raising = False)

    def failing_stage(home, *, update_args, echo):
        raise _studio_stage.StageError("clone failed")

    monkeypatch.setattr(_studio_stage, "stage", failing_stage)
    result = CliRunner().invoke(studio_mod.studio_app, ["update", "--stage"])

    assert result.exit_code == 1
    assert "[TAURI:ERROR] clone failed" in result.output


def test_update_stage_passes_the_update_options_through(monkeypatch):
    from typer.testing import CliRunner
    from unsloth_cli.commands import studio as studio_mod

    monkeypatch.delenv(_studio_stage.STAGE_ROOT_ENV, raising = False)
    captured: dict = {}

    def fake_stage(home, *, update_args, echo):
        captured["home"] = home
        captured["args"] = update_args
        return {"backend_version": "2026.9.1", "root": "/stage"}

    monkeypatch.setattr(_studio_stage, "stage", fake_stage)
    result = CliRunner().invoke(
        studio_mod.studio_app,
        ["update", "--stage", "--verbose", "--no-verify", "--package", "unsloth"],
    )

    assert result.exit_code == 0, result.output
    assert captured["home"] == studio_mod.STUDIO_HOME
    assert captured["args"] == ["--package", "unsloth", "--verbose", "--no-verify"]
    assert "Staged Unsloth Studio 2026.9.1" in result.output
