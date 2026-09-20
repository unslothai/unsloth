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


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX shebangs")
def test_make_relocatable_never_shrinks_a_script_below_its_recorded_size(tmp_path):
    """RECORD keeps the size the installer wrote and anything smaller is damage, so an 82-byte shebang would shrink every console script."""
    long_root = tmp_path / ("d" * 60) / ("e" * 60)
    long_root.mkdir(parents = True)
    venv = _make_venv(long_root)
    originals = {
        name: (venv / "bin" / name).stat().st_size
        for name in ("unsloth", "pip", "activate", "env-script", "native")
    }
    assert len(str(venv)) > 68

    assert _studio_stage.make_relocatable(venv) == 2

    for name, original in originals.items():
        assert (venv / "bin" / name).stat().st_size >= original, name
    # Padded, not truncated: the script still ends in what the installer wrote.
    text = (venv / "bin" / "unsloth").read_text(encoding = "utf-8")
    assert text.startswith(_studio_stage.RELOCATABLE_SHEBANG)
    assert text.endswith("print('cli')\n")
    assert text.splitlines()[3].startswith("# ")


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX shebangs")
def test_a_finalised_stage_under_a_long_path_passes_the_record_size_check(tmp_path):
    """End-to-end shape of the regression: the size comparison install_manifest runs after a finalised
    stage decides whether every later update repeats the whole dependency pass."""
    stage_root = tmp_path / ("l" * 70) / "studio" / _studio_stage.STAGE_DIR_NAME
    stage_root.mkdir(parents = True)
    venv = _make_venv(stage_root)
    python = venv / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    python.chmod(0o755)
    (venv / "bin" / "unsloth").chmod(0o755)
    # What RECORD holds for the console scripts, as sizes rather than a real wheel.
    recorded = {name: (venv / "bin" / name).stat().st_size for name in ("unsloth", "pip")}
    assert len(str(venv)) > 68

    _studio_stage.finalize_for_activation(stage_root)

    damaged = [
        name for name, size in recorded.items() if (venv / "bin" / name).stat().st_size < size
    ]
    assert damaged == []


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
    """An 805-807 shell still spawns `--stage`: it has to fail and leave the marker, or the shell asks again."""
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


def test_a_refusal_clears_a_stage_an_earlier_shell_left_behind(monkeypatch, tmp_path):
    """805-807 map any stage directory back to `stage`, so an orphan has the shell asking at every recheck."""
    home = tmp_path / "studio"
    stage = home / _studio_stage.STAGE_DIR_NAME
    (stage / _studio_stage.VENV_NAME / "bin").mkdir(parents = True)
    (stage / _studio_stage.VENV_NAME / "bin" / "python").write_text("x", encoding = "utf-8")
    monkeypatch.setenv(_studio_stage.SHELL_VERSION_ENV, "0.1.805-beta")

    result = _invoke_stage(monkeypatch, home)

    assert result.exit_code == 1, result.output
    assert not stage.exists()
    assert [p.name for p in home.iterdir() if p.name.startswith(".update-")] == [
        ".update-failed.json"
    ]


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("  0.1.806-beta  ", "0.1.806-beta"),
    ],
)
def test_a_refusal_records_the_shell_version_or_nothing(
    monkeypatch, tmp_path, environment, expected
):
    """`shell_version` is an `Option<String>`, so null parses; a placeholder would fail their equality check as null does."""
    home = tmp_path / "studio"
    if environment is None:
        monkeypatch.delenv(_studio_stage.SHELL_VERSION_ENV, raising = False)
    else:
        monkeypatch.setenv(_studio_stage.SHELL_VERSION_ENV, environment)

    result = _invoke_stage(monkeypatch, home)

    assert result.exit_code == 1, result.output
    marker = json.loads((home / ".update-failed.json").read_text(encoding = "utf-8"))
    assert marker["shell_version"] == expected
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
    """The 805-807 activation path imports finalize_for_activation under `python -I` inside the staged venv."""
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
