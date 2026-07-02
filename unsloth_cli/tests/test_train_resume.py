# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resume flags for `unsloth train`: registration, mutual exclusion, resolution."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# resume.py imports `utils.paths`, so the backend root must be on sys.path.
_BACKEND_ROOT = _REPO_ROOT / "studio" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _train():
    from unsloth_cli.commands import train as _train_mod
    return _train_mod.train


def _app():
    app = typer.Typer()
    app.command()(_train())
    return app


@pytest.fixture
def outputs_home(tmp_path, monkeypatch):
    # UNSLOTH_STUDIO_HOME with a fake resumable checkpoint under outputs/.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    ckpt = tmp_path / "outputs" / "checkpoint-10"
    ckpt.mkdir(parents = True)
    (ckpt / "trainer_state.json").write_text("{}", encoding = "utf-8")
    return tmp_path


def test_train_exposes_resume_options():
    params = inspect.signature(_train()).parameters
    assert "resume" in params
    assert "resume_from_checkpoint" in params


def test_resume_and_explicit_path_are_mutually_exclusive(outputs_home):
    result = CliRunner().invoke(
        _app(),
        ["--dry-run", "--resume", "--resume-from-checkpoint", "outputs/checkpoint-10"],
    )
    assert result.exit_code == 2, result.output
    assert "not both" in result.output


def test_resume_without_checkpoint_errors(tmp_path, monkeypatch):
    # Empty outputs root: bare --resume finds nothing to resume from.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    (tmp_path / "outputs").mkdir(parents = True)
    result = CliRunner().invoke(_app(), ["--dry-run", "--resume"])
    assert result.exit_code == 2, result.output
    assert "no resumable checkpoint" in result.output


def test_resume_dry_run_resolves_latest_checkpoint(outputs_home):
    result = CliRunner().invoke(_app(), ["--dry-run", "--resume"])
    assert result.exit_code == 0, result.output
    assert "resume_from_checkpoint:" in result.output
    assert "checkpoint-10" in result.output


def test_resume_from_explicit_checkpoint_path(outputs_home):
    result = CliRunner().invoke(
        _app(),
        ["--dry-run", "--resume-from-checkpoint", "outputs/checkpoint-10"],
    )
    assert result.exit_code == 0, result.output
    assert "checkpoint-10" in result.output
