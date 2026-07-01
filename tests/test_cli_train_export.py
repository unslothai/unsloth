# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for `unsloth train --export` chaining into export_checkpoint."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner


class _FakeProgress:
    error = None


class _FakeTrainer:
    """Minimal UnslothTrainer that "trains" instantly with no thread."""

    is_vlm = False
    training_thread = None

    def load_model(self, **kwargs):
        return True

    def prepare_model_for_training(self, **kwargs):
        return True

    def load_and_format_dataset(self, **kwargs):
        return ([], None)

    def start_training(self, **kwargs):
        return True

    def get_training_progress(self):
        return _FakeProgress()


def _install_fake_backend(monkeypatch: pytest.MonkeyPatch, outputs_root: Path) -> None:
    """Stub the trainer + storage-root modules the CLI lazily imports."""
    for name in (
        "studio",
        "studio.backend",
        "studio.backend.core",
        "studio.backend.core.training",
        "studio.backend.utils",
        "studio.backend.utils.paths",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    trainer_mod = types.ModuleType("studio.backend.core.training.trainer")
    trainer_mod.UnslothTrainer = _FakeTrainer
    monkeypatch.setitem(sys.modules, "studio.backend.core.training.trainer", trainer_mod)

    roots_mod = types.ModuleType("studio.backend.utils.paths.storage_roots")
    # Mirror the real resolver's "./outputs" -> outputs_root() collapse.
    roots_mod.resolve_output_dir = lambda value = None: outputs_root
    monkeypatch.setitem(sys.modules, "studio.backend.utils.paths.storage_roots", roots_mod)


@pytest.fixture
def cli_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Typer app wrapping train, with export_checkpoint patched to record args."""
    _install_fake_backend(monkeypatch, outputs_root = tmp_path / "outputs")

    from unsloth_cli.commands import export as export_cmd

    calls: list[dict] = []

    def _fake_export_checkpoint(**kwargs):
        calls.append(kwargs)
        return str(kwargs["output_dir"])

    monkeypatch.setattr(export_cmd, "export_checkpoint", _fake_export_checkpoint)

    from unsloth_cli.commands.train import train

    app = typer.Typer()
    app.command()(train)
    return app, calls, tmp_path / "outputs"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_train_export_chains_into_export_checkpoint(cli_app, runner: CliRunner) -> None:
    """`--export gguf` runs export_checkpoint against <checkpoint>/gguf by default."""
    app, calls, outputs_root = cli_app

    result = runner.invoke(
        app,
        ["--model", "hf/tiny", "--dataset", "d", "--export", "gguf"],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert len(calls) == 1
    call = calls[0]
    assert call["checkpoint"] == outputs_root
    assert call["output_dir"] == outputs_root / "gguf"
    assert call["format"] == "gguf"


def test_train_export_dir_override(cli_app, runner: CliRunner, tmp_path: Path) -> None:
    """--export-dir wins over the <checkpoint>/<format> default."""
    app, calls, _ = cli_app
    dest = tmp_path / "my-gguf"

    result = runner.invoke(
        app,
        ["--model", "hf/tiny", "--dataset", "d", "--export", "gguf", "--export-dir", str(dest)],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert calls[0]["output_dir"] == dest


def test_train_without_export_skips_export(cli_app, runner: CliRunner) -> None:
    """No --export -> export_checkpoint is never called (current behavior preserved)."""
    app, calls, _ = cli_app

    result = runner.invoke(app, ["--model", "hf/tiny", "--dataset", "d"])

    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert calls == []


def test_train_rejects_bad_export_format(cli_app, runner: CliRunner) -> None:
    """A typo'd format fails fast (exit 2) before any training happens."""
    app, calls, _ = cli_app

    result = runner.invoke(app, ["--model", "hf/tiny", "--dataset", "d", "--export", "ggu"])

    assert result.exit_code == 2
    assert "Invalid --export format" in result.output
    assert calls == []
