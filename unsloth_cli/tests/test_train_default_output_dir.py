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


class _SavingTrainer:
    is_vlm = False
    training_thread = None

    def __init__(self):
        self.output_dirs = []
        self.requested = []

    def load_model(self, **kwargs):
        return True

    def prepare_model_for_training(self, **kwargs):
        return True

    def load_and_format_dataset(self, **kwargs):
        return [], None

    def start_training(self, **kwargs):
        from utils.paths import ensure_dir, resolve_output_dir

        self.requested.append(kwargs.get("output_dir"))
        output_dir = resolve_output_dir(kwargs.get("output_dir"))
        ensure_dir(output_dir)
        (output_dir / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "unsloth/Qwen3-0.6B", "peft_type": "LORA", "r": 16}',
            encoding = "utf-8",
        )
        (output_dir / "adapter_model.safetensors").write_bytes(b"\0")
        self.output_dirs.append(output_dir)
        return True

    def get_training_progress(self):
        return SimpleNamespace(error = None, is_completed = True)


def _train(monkeypatch, trainer, *args):
    monkeypatch.setattr(train_module, "_create_cli_trainer", lambda *a: trainer)
    app = typer.Typer()
    app.command()(train_module.train)
    return CliRunner().invoke(
        app,
        ["--model", "unsloth/Qwen3-0.6B", "--dataset", "yahma/alpaca-cleaned", *args],
    )


def test_default_runs_get_their_own_listed_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    clock = iter([1000, 2000])
    monkeypatch.setattr(train_module, "time", SimpleNamespace(time = lambda: next(clock)))
    trainer = _SavingTrainer()

    assert _train(monkeypatch, trainer).exit_code == 0
    assert _train(monkeypatch, trainer).exit_code == 0

    from utils.models.checkpoints import scan_checkpoints
    from utils.paths import outputs_root

    first, second = trainer.output_dirs
    assert first != second
    assert first.parent == outputs_root() and second.parent == outputs_root()
    listed = {Path(entry[1][0][1]) for entry in scan_checkpoints()}
    assert listed == {first, second}


def test_explicit_output_dir_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    trainer = _SavingTrainer()

    assert _train(monkeypatch, trainer, "--output-dir", "my_run").exit_code == 0

    from utils.paths import outputs_root

    assert trainer.output_dirs == [outputs_root() / "my_run"]


def test_explicit_outputs_is_not_treated_as_default(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    trainer = _SavingTrainer()

    assert _train(monkeypatch, trainer, "--output-dir", "./outputs").exit_code == 0

    assert trainer.requested == ["outputs"]


def test_runs_started_in_the_same_second_get_separate_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(train_module, "time", SimpleNamespace(time = lambda: 1000))
    trainer = _SavingTrainer()

    assert _train(monkeypatch, trainer).exit_code == 0
    assert _train(monkeypatch, trainer).exit_code == 0

    first, second = trainer.output_dirs
    assert first != second
    assert first.name.endswith("_1000") and second.name.endswith("_1001")


def test_ranks_of_one_launch_share_the_launcher_folder(tmp_path, monkeypatch):
    import psutil

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Each rank finishes its model load at a different second; the launcher started once.
    clock = iter([1000, 1048])
    monkeypatch.setattr(train_module, "time", SimpleNamespace(time = lambda: next(clock)))
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(create_time = lambda: 900.5))
    trainer = _SavingTrainer()

    assert _train(monkeypatch, trainer).exit_code == 0
    assert _train(monkeypatch, trainer).exit_code == 0

    first, second = trainer.output_dirs
    assert first == second
    assert first.name.endswith("_900")
