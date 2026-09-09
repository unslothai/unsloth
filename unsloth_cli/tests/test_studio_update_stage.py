# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import platform
import subprocess
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


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX shebangs")
def test_make_relocatable_rewrites_shell_wrapper_for_path_with_spaces(tmp_path):
    venv = _make_venv(tmp_path / "stage with spaces")
    script = venv / "bin" / "pip"
    script.write_text(
        f"#!/bin/sh\n'''exec' '{venv}/bin/python' \"$0\" \"$@\"\n' '''\nprint('pip')\n",
        encoding = "utf-8",
    )

    assert _studio_stage.make_relocatable(venv) == 2

    text = script.read_text(encoding = "utf-8")
    assert text.startswith(_studio_stage.RELOCATABLE_SHEBANG)
    assert str(venv) not in text
    assert text.endswith("print('pip')\n")


def test_managed_helper_root_matches_default_and_custom_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(_studio_stage.Path, "home", lambda: tmp_path)

    assert _studio_stage.managed_helper_root(tmp_path / ".unsloth" / "studio") == (
        tmp_path / ".unsloth"
    )
    assert _studio_stage.managed_helper_root(tmp_path / "custom") == tmp_path / "custom"


def test_child_environment_points_the_staged_cli_at_the_stage_root(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRTUAL_ENV", "/elsewhere")
    monkeypatch.setenv("PYTHONHOME", "/elsewhere")
    monkeypatch.setenv("PYTHONPATH", "/foreign/checkout")

    env = _studio_stage.child_environment(tmp_path)

    assert env[_studio_stage.STAGE_ROOT_ENV] == str(tmp_path)
    assert env["PATH"].split(os.pathsep)[0] == str(
        tmp_path
        / _studio_stage.VENV_NAME
        / ("Scripts" if platform.system() == "Windows" else "bin")
    )
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env


def test_staged_python_commands_use_isolated_mode(monkeypatch, tmp_path):
    venv = _make_venv(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, env):
        commands.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(_studio_stage, "_run", fake_run)
    _studio_stage.probe_cli(venv, {})

    assert all(command[1] == "-I" for command in commands)


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX console scripts")
def test_probe_console_script_rejects_a_launcher_that_cannot_start(tmp_path):
    venv = _make_venv(tmp_path)
    launcher = venv / "bin" / "unsloth"
    launcher.write_text("#!/nonexistent/python\n", encoding = "utf-8")
    launcher.chmod(0o755)

    with pytest.raises(_studio_stage.StageError, match = "launcher"):
        _studio_stage.probe_console_script(venv, dict(os.environ))


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX console scripts")
def test_probe_console_script_accepts_a_relocated_launcher(tmp_path):
    venv = _make_venv(tmp_path)
    (venv / "bin" / "python").write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    (venv / "bin" / "python").chmod(0o755)
    _studio_stage.make_relocatable(venv)
    (venv / "bin" / "unsloth").chmod(0o755)

    _studio_stage.probe_console_script(venv, dict(os.environ))


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX console scripts")
def test_activation_finalizer_repairs_a_launcher_written_by_an_old_outer_stage(tmp_path):
    stage_root = tmp_path / "studio" / _studio_stage.STAGE_DIR_NAME
    venv = _make_venv(stage_root)
    python = venv / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    python.chmod(0o755)
    (venv / "bin" / "unsloth").chmod(0o755)

    _studio_stage.finalize_for_activation(stage_root)
    live = tmp_path / "studio" / _studio_stage.VENV_NAME
    venv.rename(live)

    result = subprocess.run([str(live / "bin" / "unsloth"), "-h"], check = False)
    assert result.returncode == 0


def _invoke_stage(monkeypatch, home: Path):
    from typer.testing import CliRunner
    from unsloth_cli.commands import studio as studio_mod

    monkeypatch.delenv(_studio_stage.STAGE_ROOT_ENV, raising = False)
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", home)
    return CliRunner().invoke(studio_mod.studio_app, ["update", "--stage"])


def test_stage_is_refused_and_records_what_the_old_shell_asked_for(monkeypatch, tmp_path):
    """An 805-807 shell still spawns `--stage`. It has to fail, and it has to leave the
    marker those shells read, or the same shell offers to prepare the update again."""
    home = tmp_path / "studio"
    monkeypatch.setenv(_studio_stage.SHELL_VERSION_ENV, "0.1.807-beta")

    result = _invoke_stage(monkeypatch, home)

    assert result.exit_code == 1, result.output
    assert "[TAURI:ERROR] background staging is no longer supported" in result.output
    marker = json.loads((home / ".update-failed.json").read_text(encoding = "utf-8"))
    # StagedVersions types this one as a plain String; null would fail the whole parse.
    assert isinstance(marker["backend_version"], str) and marker["backend_version"]
    assert marker["shell_version"] == "0.1.807-beta"
    assert not (home / _studio_stage.STAGE_DIR_NAME).exists()


def test_a_refusal_without_a_shell_version_records_a_null_one(monkeypatch, tmp_path):
    home = tmp_path / "studio"
    monkeypatch.delenv(_studio_stage.SHELL_VERSION_ENV, raising = False)

    result = _invoke_stage(monkeypatch, home)

    assert result.exit_code == 1, result.output
    marker = json.loads((home / ".update-failed.json").read_text(encoding = "utf-8"))
    assert marker["shell_version"] is None
    assert isinstance(marker["backend_version"], str)


def test_a_refusal_that_cannot_write_the_marker_still_reports_the_error(monkeypatch, tmp_path):
    # An unwritable home is the one case where the refusal matters more than the marker.
    blocker = tmp_path / "studio"
    blocker.parent.mkdir(parents = True, exist_ok = True)
    blocker.write_text("not a directory", encoding = "utf-8")

    result = _invoke_stage(monkeypatch, blocker)

    assert result.exit_code == 1, result.output
    assert "[TAURI:ERROR] background staging is no longer supported" in result.output


def test_the_activation_finalizer_imports_under_isolated_python():
    """The 805-807 activation path runs `python -I -c "from unsloth_cli._studio_stage
    import finalize_for_activation"` inside the staged venv. Any import this module
    grows beyond the stdlib breaks that, and only there."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from unsloth_cli._studio_stage import finalize_for_activation",
            str(_REPO_ROOT),
        ],
        capture_output = True,
        text = True,
        check = False,
    )

    assert result.returncode == 0, result.stderr
