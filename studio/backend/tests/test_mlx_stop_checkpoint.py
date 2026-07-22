# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for MLX stop-and-save checkpoint handling."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


_BACKEND = Path(__file__).resolve().parents[1]


def _load_worker_module():
    spec = importlib.util.spec_from_file_location(
        "training_worker_under_test",
        _BACKEND / "core" / "training" / "worker.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


worker = _load_worker_module()


class _FakeTrainer:
    def __init__(self, step: int):
        self._global_step = step
        self._train_loss_history = []
        self.model = object()


def _write_checkpoint(out: Path, step: int) -> Path:
    checkpoint = out / f"checkpoint-{step}"
    checkpoint.mkdir(parents = True, exist_ok = True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding = "utf-8"
    )
    save_file({"weight": np.ones(1, dtype = np.float32)}, checkpoint / "adapters.safetensors")
    save_file(
        {"state": np.ones(1, dtype = np.float32)},
        checkpoint / "optimizer_state.safetensors",
    )
    return checkpoint


def test_mlx_has_checkpoint_at_step_requires_complete_state(tmp_path):
    out = tmp_path / "outputs" / "run_x"
    _write_checkpoint(out, 5)

    assert worker._mlx_has_checkpoint_at_step(out, 5) is True


def test_write_mlx_stop_checkpoint_returns_true_when_current_step_checkpoint_exists(tmp_path):
    out = tmp_path / "outputs" / "run_x"
    _write_checkpoint(out, 5)

    assert worker._write_mlx_stop_checkpoint(_FakeTrainer(step = 5), object(), out) is True


def test_write_mlx_stop_checkpoint_writes_current_step_when_only_older_checkpoint_exists(
    tmp_path, monkeypatch
):
    out = tmp_path / "outputs" / "run_x"
    _write_checkpoint(out, 5)

    saved_steps: list[int] = []

    def _save_state(_value, path, name):
        save_file({"state": np.ones(1, dtype = np.float32)}, Path(path, name))

    def _save_trainer_state(state, ckpt_dir, **_kwargs):
        Path(ckpt_dir, "trainer_state.json").write_text(json.dumps(state), encoding = "utf-8")
        saved_steps.append(int(state["global_step"]))

    fake_utils = types.SimpleNamespace(
        save_trainable_adapters = lambda model, path: _save_state(
            model, path, "adapters.safetensors"
        ),
        save_optimizer_state = lambda optimizer, path: _save_state(
            optimizer, path, "optimizer_state.safetensors"
        ),
        save_trainer_state = _save_trainer_state,
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", fake_utils)

    assert worker._write_mlx_stop_checkpoint(_FakeTrainer(step = 10), object(), out) is True
    assert saved_steps == [10]
    assert (out / "checkpoint-10" / "trainer_state.json").is_file()


def test_write_mlx_stop_checkpoint_returns_false_without_optimizer(tmp_path):
    out = tmp_path / "outputs" / "run_x"
    out.mkdir(parents = True)

    assert worker._write_mlx_stop_checkpoint(_FakeTrainer(step = 5), None, out) is False


def test_write_mlx_stop_checkpoint_rejects_incomplete_current_checkpoint(tmp_path):
    out = tmp_path / "outputs" / "run_x"
    ckpt = out / "checkpoint-5"
    ckpt.mkdir(parents = True)
    (ckpt / "trainer_state.json").write_text('{"global_step": 5}', encoding = "utf-8")

    assert worker._write_mlx_stop_checkpoint(_FakeTrainer(step = 5), None, out) is False


def test_write_mlx_stop_checkpoint_ignores_stale_checkpoint_without_optimizer(tmp_path):
    # An older checkpoint does not cover the current step, so this still fails.
    out = tmp_path / "outputs" / "run_x"
    _write_checkpoint(out, 5)

    assert worker._write_mlx_stop_checkpoint(_FakeTrainer(step = 10), None, out) is False


def test_write_mlx_stop_checkpoint_returns_false_when_save_fails(tmp_path, monkeypatch):
    out = tmp_path / "outputs" / "run_x"
    out.mkdir(parents = True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("save failed")

    fake_utils = types.SimpleNamespace(
        save_trainable_adapters = _boom,
        save_optimizer_state = lambda *_a, **_k: None,
        save_trainer_state = lambda *_a, **_k: None,
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", fake_utils)

    assert worker._write_mlx_stop_checkpoint(_FakeTrainer(step = 5), object(), out) is False
