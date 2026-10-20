# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.train as train_module  # noqa: E402
from unsloth_cli.config import Config  # noqa: E402


class _RecordingTrainer:
    is_vlm = False
    training_thread = None

    def __init__(self):
        self.prepare_kwargs = None

    def load_model(self, **kwargs):
        return True

    def prepare_model_for_training(self, **kwargs):
        self.prepare_kwargs = kwargs
        return True

    def load_and_format_dataset(self, **kwargs):
        return [], None

    def start_training(self, **kwargs):
        return True

    def get_training_progress(self):
        return SimpleNamespace(error = None, is_completed = True)


def test_random_seed_flag_seeds_the_lora_init(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "model: unsloth/Qwen3-0.6B\ndata:\n  dataset: yahma/alpaca-cleaned\n",
        encoding = "utf-8",
    )
    trainer = _RecordingTrainer()
    monkeypatch.setattr(train_module, "_create_cli_trainer", lambda *args: trainer)
    app = typer.Typer()
    app.command()(train_module.train)

    result = CliRunner().invoke(app, ["--config", str(config), "--random-seed", "42"])

    assert result.exit_code == 0, result.output
    assert trainer.prepare_kwargs["random_state"] == 42


def test_mlx_adapter_accepts_the_cli_lora_kwargs():
    backend = str(_REPO_ROOT / "studio" / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from core.training.training import create_mlx_trainer_adapter

    adapter = create_mlx_trainer_adapter()
    adapter._model_config = {}

    assert adapter.prepare_model_for_training(**Config().model_kwargs(True, False))
