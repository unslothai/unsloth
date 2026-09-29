# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli.commands.train as train_module  # noqa: E402


class _FakeThread:
    def __init__(self, trainer):
        self._trainer = trainer

    def is_alive(self):
        return not self._trainer.saved

    def join(self, timeout = None):
        self._trainer.finish()


class _FakeTrainer:
    is_vlm = False

    def __init__(self, save_error = None):
        self.progress = SimpleNamespace(error = None, is_completed = False)
        self.training_thread = None
        self.stopped = False
        self.saved = False
        self._save_error = save_error

    def load_model(self, **kwargs):
        return True

    def prepare_model_for_training(self, **kwargs):
        return True

    def load_and_format_dataset(self, **kwargs):
        return [], None

    def start_training(self, **kwargs):
        self.training_thread = _FakeThread(self)
        return True

    def stop_training(self, save = True):
        self.stopped = True

    def get_training_progress(self):
        return self.progress

    def finish(self):
        if self.saved:
            return
        self.saved = True
        if self._save_error:
            self.progress.error = self._save_error
        else:
            self.progress.is_completed = not self.stopped


def _run(tmp_path, monkeypatch, trainer, sleep):
    config = tmp_path / "config.yaml"
    config.write_text(
        "model: unsloth/Qwen3-0.6B\ndata:\n  dataset: yahma/alpaca-cleaned\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(train_module, "_create_cli_trainer", lambda *args: trainer)
    monkeypatch.setattr(train_module, "time", SimpleNamespace(sleep = sleep))
    app = typer.Typer()
    app.command()(train_module.train)
    return CliRunner().invoke(app, ["--config", str(config)])


def _interrupt(seconds):
    raise KeyboardInterrupt


def test_natural_completion_exits_zero(tmp_path, monkeypatch):
    trainer = _FakeTrainer()

    result = _run(tmp_path, monkeypatch, trainer, lambda seconds: trainer.finish())

    assert result.exit_code == 0
    assert not trainer.stopped


def test_ctrl_c_saves_then_exits_130(tmp_path, monkeypatch):
    trainer = _FakeTrainer()

    result = _run(tmp_path, monkeypatch, trainer, _interrupt)

    assert result.exit_code == 130
    assert trainer.stopped
    assert trainer.saved
    assert "Stopping training (Ctrl+C detected)" in result.output


def test_ctrl_c_after_training_already_completed_exits_zero(tmp_path, monkeypatch):
    trainer = _FakeTrainer()

    def finish_then_interrupt(seconds):
        trainer.finish()
        raise KeyboardInterrupt

    result = _run(tmp_path, monkeypatch, trainer, finish_then_interrupt)

    assert result.exit_code == 0
    assert trainer.progress.is_completed


def test_ctrl_c_with_a_failed_save_still_reports_the_error(tmp_path, monkeypatch):
    trainer = _FakeTrainer(save_error = "disk full")

    result = _run(tmp_path, monkeypatch, trainer, _interrupt)

    assert result.exit_code == 1
    assert "Training error: disk full" in result.output


def _fake_mlx_worker(config, event_queue, stop_queue):
    try:
        stop_queue.get(timeout = 0.5)
        status = "Training stopped"
    except queue.Empty:
        status = "Training completed"
    event_queue.put({"type": "complete", "output_dir": "out", "status_message": status})


@pytest.fixture
def mlx_adapter():
    backend = str(_REPO_ROOT / "studio" / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from core.training.training import create_mlx_trainer_adapter

    adapter = create_mlx_trainer_adapter()
    adapter.load_model = lambda **kwargs: True
    adapter.prepare_model_for_training = lambda **kwargs: True
    adapter.load_and_format_dataset = lambda **kwargs: ([], None)
    adapter._model_config = {"model_name": "unsloth/Qwen3-0.6B"}
    adapter._dataset_config = {"dataset_source": "yahma/alpaca-cleaned"}
    adapter._build_worker_config = lambda training_args: {}
    adapter._run_mlx_worker = _fake_mlx_worker
    return adapter


def test_mlx_adapter_ctrl_c_exits_130(tmp_path, monkeypatch, mlx_adapter):
    result = _run(tmp_path, monkeypatch, mlx_adapter, _interrupt)

    assert result.exit_code == 130
    assert mlx_adapter.get_training_progress().status_message == "Training stopped"


def test_mlx_adapter_natural_completion_exits_zero(tmp_path, monkeypatch, mlx_adapter):
    result = _run(tmp_path, monkeypatch, mlx_adapter, time.sleep)

    assert result.exit_code == 0
    assert mlx_adapter.get_training_progress().is_completed
