# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import typer


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _missing_structlog():
    raise ModuleNotFoundError("No module named 'structlog'", name = "structlog")


def test_desktop_runtime_check_reports_missing_dependency_as_json(monkeypatch, capsys):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    monkeypatch.setattr(studio, "_load_run_module", _missing_structlog)

    with pytest.raises(typer.Exit) as exited:
        studio.desktop_runtime_check(_json_output = True)

    assert exited.value.exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "runtime_ready": False,
        "reason": "missing_dependency",
        "module": "structlog",
    }


def test_desktop_runtime_check_reports_success(monkeypatch, capsys):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    monkeypatch.setattr(studio, "_load_run_module", lambda: object())
    monkeypatch.setattr(studio, "_missing_studio_requirement", lambda _run_mod: None)

    studio.desktop_runtime_check(_json_output = True)

    assert json.loads(capsys.readouterr().out) == {"runtime_ready": True}


def test_desktop_runtime_check_catches_later_startup_dependency(monkeypatch, capsys, tmp_path):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text(
        "definitely-missing-studio-package\n",
        encoding = "utf-8",
    )
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)

    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["module"] == "definitely-missing-studio-package"


def test_desktop_runtime_check_rejects_version_mismatch(monkeypatch, capsys, tmp_path):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text("example-package==2.0\n", encoding = "utf-8")
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)
    monkeypatch.setattr(
        importlib.import_module("importlib.metadata"),
        "distribution",
        lambda _name: SimpleNamespace(version = "1.0"),
    )

    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["module"] == "example-package"
