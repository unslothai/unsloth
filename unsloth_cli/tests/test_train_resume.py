# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resume flags for `unsloth train`: registration, mutual exclusion, resolution."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch
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


def _write_checkpoint(out: Path, step: int) -> Path:
    # A full bundle: resume validation rejects a bare trainer_state.json.
    checkpoint = out / f"checkpoint-{step}"
    checkpoint.mkdir(parents = True, exist_ok = True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding = "utf-8"
    )
    torch.save({"weight": torch.ones(1)}, checkpoint / "adapter_model.bin")
    torch.save({"state": {0: torch.ones(1)}}, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": step}, checkpoint / "scheduler.pt")
    return checkpoint


@pytest.fixture(autouse = True)
def _pytorch_backend(monkeypatch):
    # Host-independent default: the MLX-specific tests override explicitly.
    from unsloth_cli.commands import train as train_cmd
    monkeypatch.setattr(train_cmd, "_should_use_mlx_backend_for_cli", lambda: False)


@pytest.fixture
def outputs_home(tmp_path, monkeypatch):
    # UNSLOTH_STUDIO_HOME with a fake resumable checkpoint under outputs/.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    _write_checkpoint(tmp_path / "outputs", 10)
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


def test_resume_from_outside_outputs_root_explains_constraint(outputs_home, tmp_path_factory):
    # The trainer only writes under the outputs root; the error must say so.
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    result = CliRunner().invoke(
        _app(),
        ["--dry-run", "--resume-from-checkpoint", str(elsewhere)],
    )
    assert result.exit_code == 2, result.output
    assert "write checkpoints under" in result.output


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


def test_resume_from_external_mlx_output_dir(outputs_home, tmp_path_factory, monkeypatch):
    # The MLX CLI adapter writes checkpoints under a cwd-absolutized dir the
    # outputs-root helpers cannot see; a valid checkpoint there must resume.
    from unsloth_cli.commands import train as train_cmd

    monkeypatch.setattr(train_cmd, "_should_use_mlx_backend_for_cli", lambda: True)
    external = tmp_path_factory.mktemp("mlx_run")
    import numpy as np
    from safetensors.numpy import save_file

    ckpt = external / "checkpoint-25"
    ckpt.mkdir(parents = True)
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 25}))
    save_file({"weight": np.ones(1, dtype = np.float32)}, ckpt / "adapters.safetensors")
    save_file({"state": np.ones(1, dtype = np.float32)}, ckpt / "optimizer_state.safetensors")
    result = CliRunner().invoke(
        _app(),
        ["--dry-run", "--resume-from-checkpoint", str(external)],
    )
    assert result.exit_code == 0, result.output
    assert "checkpoint-25" in result.output


def test_external_fallback_is_mlx_only(outputs_home, tmp_path_factory, monkeypatch):
    # On the PyTorch path the trainer rejects external output dirs at training
    # time, so a dry run must not accept them either.
    from unsloth_cli.commands import train as train_cmd

    monkeypatch.setattr(train_cmd, "_should_use_mlx_backend_for_cli", lambda: False)
    external = tmp_path_factory.mktemp("mlx_run2")
    _write_checkpoint(external, 30)
    result = CliRunner().invoke(
        _app(),
        ["--dry-run", "--resume-from-checkpoint", str(external)],
    )
    assert result.exit_code == 2, result.output
    assert "write checkpoints under" in result.output


def test_pytorch_resume_rejects_mlx_only_checkpoints(outputs_home, monkeypatch):
    # An MLX bundle under the outputs root must not pass a PyTorch dry run
    # that would fail only after model loading.
    import json as _json
    from unsloth_cli.commands import train as train_cmd

    monkeypatch.setattr(train_cmd, "_should_use_mlx_backend_for_cli", lambda: False)
    mlx_ckpt = outputs_home / "outputs" / "checkpoint-40"
    mlx_ckpt.mkdir(parents = True)
    import numpy as np
    from safetensors.numpy import save_file

    (mlx_ckpt / "trainer_state.json").write_text(_json.dumps({"global_step": 40}))
    save_file({"weight": np.ones(1, dtype = np.float32)}, mlx_ckpt / "adapters.safetensors")
    save_file({"state": np.ones(1, dtype = np.float32)}, mlx_ckpt / "optimizer_state.safetensors")
    result = CliRunner().invoke(
        _app(),
        ["--dry-run", "--resume-from-checkpoint", "outputs/checkpoint-40"],
    )
    assert result.exit_code == 2, result.output
