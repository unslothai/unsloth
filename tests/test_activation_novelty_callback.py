"""Unit tests for ActivationNoveltyCallback.

Pure CPU tests — no GPU, no pretrained weights, no network access.
Only torch and transformers are required.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_callback():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from unsloth.callbacks import ActivationNoveltyCallback
    return ActivationNoveltyCallback


def _make_state_control():
    from transformers import TrainerControl, TrainerState, TrainingArguments

    args = TrainingArguments(output_dir="/tmp/_test_ncg_cb", no_cuda=True)
    state = TrainerState()
    state.global_step = 0
    control = TrainerControl()
    return args, state, control


class _TinyMLP(nn.Module):
    def __init__(self, d: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return self.fc(x)


class _TinyModel(nn.Module):
    """Minimal model whose sub-module name matches the 'mlp' search key."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TinyMLP(16)

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_default_instantiation():
    cb = _import_callback()()
    assert cb.layer_idx == -1
    assert cb.early_stop is False
    assert cb.log_key == "activation_novelty"
    assert cb.novelty_threshold == pytest.approx(0.1)
    assert cb.window == 3


def test_custom_params():
    cb = _import_callback()(
        layer_idx=2,
        novelty_threshold=0.2,
        window=5,
        early_stop=True,
        log_key="my_novelty",
    )
    assert cb.layer_idx == 2
    assert cb.window == 5
    assert cb.early_stop is True
    assert cb.log_key == "my_novelty"


# ---------------------------------------------------------------------------
# Novelty computation
# ---------------------------------------------------------------------------

def test_compute_novelty_uniform():
    """Uniform activations -> max novelty (near 1.0)."""
    cb = _import_callback()()
    d = 64
    cb._activations = [torch.ones(8, d)]
    novelty = cb._compute_novelty()
    assert novelty > 0.99


def test_compute_novelty_spike():
    """Single active neuron -> near-zero novelty."""
    cb = _import_callback()()
    d = 64
    x = torch.zeros(8, d)
    x[:, 0] = 1.0
    cb._activations = [x]
    novelty = cb._compute_novelty()
    assert novelty < 0.05


def test_compute_novelty_empty_returns_last():
    cb = _import_callback()()
    cb._last_novelty = 0.42
    assert cb._compute_novelty() == pytest.approx(0.42)


def test_compute_novelty_clipped_to_unit_interval():
    cb = _import_callback()()
    cb._activations = [torch.ones(4, 4)]
    novelty = cb._compute_novelty()
    assert 0.0 <= novelty <= 1.0


# ---------------------------------------------------------------------------
# Hook registration and activation capture
# ---------------------------------------------------------------------------

def test_hook_captures_activations():
    cb = _import_callback()(layer_getter=lambda m: m.mlp)
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model=model)
    assert cb._handle is not None

    _ = model(torch.randn(4, 16))
    assert len(cb._activations) == 1
    assert cb._activations[0].shape[1] == 16

    cb.on_train_end(args, state, control)
    assert cb._handle is None


def test_hook_removed_after_train_end():
    cb = _import_callback()(layer_getter=lambda m: m.mlp)
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model=model)
    cb.on_train_end(args, state, control)

    # After removal, forward pass must not populate _activations.
    _ = model(torch.randn(4, 16))
    assert len(cb._activations) == 0


def test_no_hook_without_model():
    cb = _import_callback()()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model=None)
    assert cb._handle is None


def test_auto_layer_detection():
    """_find_mlp_layers should locate the 'mlp' sub-module automatically."""
    cb = _import_callback()()   # no layer_getter
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model=model)
    assert cb._handle is not None
    cb.on_train_end(args, state, control)


# ---------------------------------------------------------------------------
# on_evaluate
# ---------------------------------------------------------------------------

def test_on_evaluate_clears_activations():
    cb = _import_callback()(layer_getter=lambda m: m.mlp)
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model=model)
    _ = model(torch.randn(4, 16))
    assert len(cb._activations) == 1

    cb.on_evaluate(args, state, control, model=model)
    assert len(cb._activations) == 0


def test_on_evaluate_updates_history():
    cb = _import_callback()(layer_getter=lambda m: m.mlp, window=3)
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model=model)

    for _ in range(2):
        _ = model(torch.randn(4, 16))
        cb.on_evaluate(args, state, control, model=model)

    assert len(cb._history) == 2


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def test_early_stop_triggers_after_window():
    cb = _import_callback()(
        layer_getter=lambda m: m.mlp,
        novelty_threshold=0.5,
        window=2,
        early_stop=True,
    )
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model=model)

    # Inject spike activations -> novelty << threshold
    for _ in range(2):
        x = torch.zeros(4, 16)
        x[:, 0] = 1.0
        cb._activations = [x]
        control = cb.on_evaluate(args, state, control, model=model)

    assert control.should_training_stop is True


def test_early_stop_not_triggered_before_window():
    cb = _import_callback()(
        layer_getter=lambda m: m.mlp,
        novelty_threshold=0.5,
        window=3,
        early_stop=True,
    )
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model=model)

    # Only 2 evals, window=3 -> should not stop yet
    for _ in range(2):
        x = torch.zeros(4, 16)
        x[:, 0] = 1.0
        cb._activations = [x]
        control = cb.on_evaluate(args, state, control, model=model)

    assert not control.should_training_stop


def test_early_stop_false_never_stops():
    cb = _import_callback()(
        layer_getter=lambda m: m.mlp,
        novelty_threshold=0.99,
        window=1,
        early_stop=False,
    )
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model=model)

    x = torch.zeros(4, 16)
    x[:, 0] = 1.0
    cb._activations = [x]
    control = cb.on_evaluate(args, state, control, model=model)

    assert not control.should_training_stop


def test_early_stop_does_not_trigger_on_high_novelty():
    cb = _import_callback()(
        layer_getter=lambda m: m.mlp,
        novelty_threshold=0.05,
        window=2,
        early_stop=True,
    )
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model=model)

    # Uniform activations -> high novelty, should not stop
    for _ in range(3):
        cb._activations = [torch.ones(4, 16)]
        control = cb.on_evaluate(args, state, control, model=model)

    assert not control.should_training_stop


# ---------------------------------------------------------------------------
# on_log
# ---------------------------------------------------------------------------

def test_on_log_injects_metric():
    cb = _import_callback()()
    cb._last_novelty = 0.77
    logs = {"loss": 1.5}
    args, state, control = _make_state_control()
    cb.on_log(args, state, control, logs=logs)
    assert logs["activation_novelty"] == pytest.approx(0.77)


def test_on_log_custom_key():
    cb = _import_callback()(log_key="rep_novelty")
    cb._last_novelty = 0.33
    logs = {}
    args, state, control = _make_state_control()
    cb.on_log(args, state, control, logs=logs)
    assert "rep_novelty" in logs
    assert logs["rep_novelty"] == pytest.approx(0.33)


def test_on_log_no_crash_when_logs_none():
    cb = _import_callback()()
    args, state, control = _make_state_control()
    cb.on_log(args, state, control, logs=None)   # must not raise
