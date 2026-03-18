import os
import sys
from pathlib import Path

import pytest
import typer

from unsloth_cli.commands import studio


def test_reexec_cli_in_studio_venv_execs_on_unix(monkeypatch):
    monkeypatch.setattr(studio, "STUDIO_HOME", Path("/tmp/unsloth-studio"))
    monkeypatch.setattr(sys, "prefix", "/usr/local")
    monkeypatch.setattr(
        studio,
        "_studio_venv_python",
        lambda: Path("/tmp/unsloth-studio/.venv/bin/python"),
    )

    captured = {}

    def fake_execvp(executable, args):
        captured["executable"] = executable
        captured["args"] = args
        raise SystemExit(0)

    monkeypatch.setattr(os, "execvp", fake_execvp)
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(SystemExit) as excinfo:
        studio._reexec_cli_in_studio_venv(["train", "--help"], silent = True)

    assert excinfo.value.code == 0
    assert captured["executable"] == "/tmp/unsloth-studio/.venv/bin/python"
    assert captured["args"] == [
        "/tmp/unsloth-studio/.venv/bin/python",
        "-m",
        "unsloth_cli",
        "train",
        "--help",
    ]


def test_reexec_cli_in_studio_venv_noops_inside_venv(monkeypatch):
    monkeypatch.setattr(studio, "STUDIO_HOME", Path("/tmp/unsloth-studio"))
    monkeypatch.setattr(sys, "prefix", "/tmp/unsloth-studio/.venv")

    called = {"value": False}

    def fake_studio_python():
        called["value"] = True
        return Path("/tmp/unsloth-studio/.venv/bin/python")

    monkeypatch.setattr(studio, "_studio_venv_python", fake_studio_python)

    studio._reexec_cli_in_studio_venv(["inference", "model", "prompt"], silent = True)

    assert called["value"] is False


def test_reexec_cli_in_studio_venv_requires_setup(monkeypatch):
    monkeypatch.setattr(studio, "STUDIO_HOME", Path("/tmp/unsloth-studio"))
    monkeypatch.setattr(sys, "prefix", "/usr/local")
    monkeypatch.setattr(studio, "_studio_venv_python", lambda: None)

    with pytest.raises(typer.Exit) as excinfo:
        studio._reexec_cli_in_studio_venv(["export", "ckpt", "out"], silent = True)

    assert excinfo.value.exit_code == 1


def test_stage_setup_script_only_when_inside_studio_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path / ".unsloth" / "studio")

    outside_root = tmp_path / "repo" / "studio"
    outside_root.mkdir(parents = True)
    outside_script = outside_root / "setup.sh"
    outside_script.write_text("#!/usr/bin/env bash\n", encoding = "utf-8")

    staged_script, temp_dir = studio._stage_setup_script_if_needed(outside_script)

    assert staged_script == outside_script.resolve()
    assert temp_dir is None


def test_stage_setup_script_copies_when_inside_studio_venv(monkeypatch, tmp_path):
    studio_home = tmp_path / ".unsloth" / "studio"
    monkeypatch.setattr(studio, "STUDIO_HOME", studio_home)

    package_root = (
        studio_home / ".venv" / "lib" / "python3.11" / "site-packages" / "studio"
    )
    package_root.mkdir(parents = True)
    script = package_root / "setup.sh"
    helper = package_root / "install_python_stack.py"
    script.write_text("#!/usr/bin/env bash\n", encoding = "utf-8")
    helper.write_text("print('ok')\n", encoding = "utf-8")

    staged_script, temp_dir = studio._stage_setup_script_if_needed(script)
    try:
        assert temp_dir is not None
        assert staged_script != script.resolve()
        assert staged_script.name == "setup.sh"
        assert staged_script.is_file()
        assert (staged_script.parent / "install_python_stack.py").is_file()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
