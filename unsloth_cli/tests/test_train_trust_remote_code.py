# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.train as train_module  # noqa: E402


class _RecordingTrainer:
    is_vlm = False
    training_thread = None

    def __init__(self):
        self.load_kwargs = None

    def load_model(self, **kwargs):
        self.load_kwargs = kwargs
        return True

    def prepare_model_for_training(self, **kwargs):
        return True

    def load_and_format_dataset(self, **kwargs):
        return [], None

    def start_training(self, **kwargs):
        return True

    def get_training_progress(self):
        return SimpleNamespace(error = None, is_completed = True)


@pytest.mark.parametrize(
    "yaml_extra, argv, expected",
    [
        ("", [], False),
        ("", ["--trust-remote-code"], True),
        ("training:\n  trust_remote_code: true\n", [], True),
        ("training:\n  trust_remote_code: true\n", ["--no-trust-remote-code"], False),
    ],
)
def test_train_forwards_trust_remote_code_to_load_model(
    tmp_path, monkeypatch, yaml_extra, argv, expected
):
    config = tmp_path / "config.yaml"
    config.write_text(
        "model: org/custom-code-model\ndata:\n  dataset: yahma/alpaca-cleaned\n" + yaml_extra,
        encoding = "utf-8",
    )
    trainer = _RecordingTrainer()
    monkeypatch.setattr(train_module, "_create_cli_trainer", lambda *args: trainer)
    monkeypatch.setattr(
        train_module, "_dataset_modalities", lambda *args: (False, False), raising = False
    )
    app = typer.Typer()
    app.command()(train_module.train)

    result = CliRunner().invoke(app, ["--config", str(config), *argv])

    assert result.exit_code == 0, result.output
    assert trainer.load_kwargs["trust_remote_code"] is expected
