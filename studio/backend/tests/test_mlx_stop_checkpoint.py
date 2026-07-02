# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for MLX stop-and-save checkpoint handling."""

import importlib.util
import sys
import types
from pathlib import Path


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


def test_mlx_output_has_resume_checkpoint_detects_trainer_state(tmp_path):
    out = tmp_path / "outputs" / "run_x"
    ckpt = out / "checkpoint-5"
    ckpt.mkdir(parents = True)
    (ckpt / "trainer_state.json").write_text("{}", encoding = "utf-8")

    assert worker._mlx_output_has_resume_checkpoint(out) is True


def test_write_mlx_stop_checkpoint_returns_true_when_current_step_checkpoint_exists(tmp_path):
    out = tmp_path / "outputs" / "run_x"
    ckpt = out / "checkpoint-5"
    ckpt.mkdir(parents = True)
    (ckpt / "trainer_state.json").write_text("{}", encoding = "utf-8")

    assert worker._write_mlx_stop_checkpoint(_FakeTrainer(step = 5), object(), out) is True


def test_write_mlx_stop_checkpoint_writes_current_step_when_only_older_checkpoint_exists(
    tmp_path, monkeypatch
):
    out = tmp_path / "outputs" / "run_x"
    old_ckpt = out / "checkpoint-5"
    old_ckpt.mkdir(parents = True)
    (old_ckpt / "trainer_state.json").write_text("{}", encoding = "utf-8")

    saved_steps: list[int] = []

    def _save_trainer_state(state, ckpt_dir, **_kwargs):
        Path(ckpt_dir, "trainer_state.json").write_text("{}", encoding = "utf-8")
        saved_steps.append(int(state["global_step"]))

    fake_utils = types.SimpleNamespace(
        save_trainable_adapters = lambda *_a, **_k: None,
        save_optimizer_state = lambda *_a, **_k: None,
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


def test_write_mlx_stop_checkpoint_accepts_existing_checkpoint_without_optimizer(tmp_path):
    # A checkpoint at the current step counts even when no optimizer was captured.
    out = tmp_path / "outputs" / "run_x"
    ckpt = out / "checkpoint-5"
    ckpt.mkdir(parents = True)
    (ckpt / "trainer_state.json").write_text("{}", encoding = "utf-8")

    assert worker._write_mlx_stop_checkpoint(_FakeTrainer(step = 5), None, out) is True


def test_write_mlx_stop_checkpoint_ignores_stale_checkpoint_without_optimizer(tmp_path):
    # An older checkpoint does not cover the current step, so this still fails.
    out = tmp_path / "outputs" / "run_x"
    ckpt = out / "checkpoint-5"
    ckpt.mkdir(parents = True)
    (ckpt / "trainer_state.json").write_text("{}", encoding = "utf-8")

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


def test_worker_sigint_guard_survives_first_interrupt_only(monkeypatch):
    # First Ctrl+C is the parent's stop-and-save window; the second force-quits.
    import signal as signal_mod

    installed = {}
    monkeypatch.setattr(signal_mod, "signal", lambda s, h: installed.setdefault(s, h))
    exits: list = []
    monkeypatch.setattr(worker.os, "_exit", lambda code: exits.append(code))

    worker._install_worker_sigint_guard()
    handler = installed[signal_mod.SIGINT]

    handler(signal_mod.SIGINT, None)
    assert exits == []

    handler(signal_mod.SIGINT, None)
    assert exits == [130]
