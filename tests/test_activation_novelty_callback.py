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

    args = TrainingArguments(output_dir = "/tmp/_test_ncg_cb", no_cuda = True)
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


class _TinySeqMLP(nn.Module):
    """MLP that accepts (batch, seq, hidden) and returns the same shape."""

    def __init__(self, d: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):  # x: (batch, seq, d)
        return self.fc(x)  # (batch, seq, d)


class _TinySeqModel(nn.Module):
    """Model whose MLP outputs 3-D tensors (batch, seq, hidden)."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TinySeqMLP(16)

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
        layer_idx = 2,
        novelty_threshold = 0.2,
        window = 5,
        early_stop = True,
        log_key = "my_novelty",
    )
    assert cb.layer_idx == 2
    assert cb.window == 5
    assert cb.early_stop is True
    assert cb.log_key == "my_novelty"


def test_window_zero_raises():
    with pytest.raises(ValueError, match = "window"):
        _import_callback()(window = 0)


def test_window_negative_raises():
    with pytest.raises(ValueError, match = "window"):
        _import_callback()(window = -1)


def test_layer_idx_out_of_range_does_not_crash():
    """An out-of-range layer_idx should print a warning and skip, not raise."""
    cb = _import_callback()(layer_idx = 99)
    model = _TinyModel()  # has exactly 1 MLP layer
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)
    assert cb._handle is None


# ---------------------------------------------------------------------------
# Novelty computation
# ---------------------------------------------------------------------------


def _inject_mean_abs(cb, tensor_2d: torch.Tensor) -> None:
    """Simulate what the hook would compute for a 2-D (batch, hidden) tensor."""
    cb._running_mean_abs = tensor_2d.abs().mean(dim = 0)
    cb._running_count = 1


def test_compute_novelty_uniform():
    """Uniform activations -> max novelty (near 1.0)."""
    cb = _import_callback()()
    _inject_mean_abs(cb, torch.ones(8, 64))
    assert cb._compute_novelty() > 0.99


def test_compute_novelty_spike():
    """Single active neuron -> near-zero novelty."""
    cb = _import_callback()()
    x = torch.zeros(8, 64)
    x[:, 0] = 1.0
    _inject_mean_abs(cb, x)
    assert cb._compute_novelty() < 0.05


def test_compute_novelty_empty_returns_last():
    cb = _import_callback()()
    cb._last_novelty = 0.42
    assert cb._compute_novelty() == pytest.approx(0.42)


def test_compute_novelty_clipped_to_unit_interval():
    cb = _import_callback()()
    _inject_mean_abs(cb, torch.ones(4, 4))
    assert 0.0 <= cb._compute_novelty() <= 1.0


# ---------------------------------------------------------------------------
# Hook registration and activation capture
# ---------------------------------------------------------------------------


def test_hook_captures_activations():
    cb = _import_callback()(layer_getter = lambda m: m.mlp)
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model = model)
    assert cb._handle is not None

    # Hook captures eval-phase activations only.
    model.eval()
    _ = model(torch.randn(4, 16))
    # Running stats should be populated; shape is (hidden,) = (16,)
    assert cb._running_mean_abs is not None
    assert cb._running_mean_abs.shape == (16,)
    assert cb._running_count == 1

    cb.on_train_end(args, state, control)
    assert cb._handle is None


def test_hook_removed_after_train_end():
    cb = _import_callback()(layer_getter = lambda m: m.mlp)
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model = model)
    cb.on_train_end(args, state, control)

    # After removal, eval-mode forward pass must not update running stats.
    model.eval()
    _ = model(torch.randn(4, 16))
    assert cb._running_mean_abs is None


def test_no_hook_without_model():
    cb = _import_callback()()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = None)
    assert cb._handle is None


def test_hook_ignores_training_mode_activations():
    """Hook must not accumulate stats during training steps; only eval-phase."""
    cb = _import_callback()(layer_getter = lambda m: m.mlp)
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model = model)

    # Training forward pass — must be ignored.
    model.train()
    _ = model(torch.randn(4, 16))
    assert cb._running_mean_abs is None
    assert cb._running_count == 0

    # Eval forward pass — must be captured.
    model.eval()
    _ = model(torch.randn(4, 16))
    assert cb._running_mean_abs is not None
    cb.on_train_end(args, state, control)


def test_hook_3d_abs_before_sequence_mean():
    """For 3-D MLP outputs, abs() must be applied before seq-axis mean.

    A neuron that fires +1 on token 0 and -1 on token 1 is active; the signed
    mean is 0 which would make it look silent.  The hook must take abs first.
    """
    cb = _import_callback()(layer_getter = lambda m: m.mlp)
    model = _TinySeqModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)

    # Zero-weight model so output == input; craft input where token-level
    # signed values cancel: token 0 = +1, token 1 = -1 for every neuron.
    with torch.no_grad():
        model.mlp.fc.weight.copy_(torch.eye(16))
        model.mlp.fc.bias.zero_()

    x = torch.zeros(1, 2, 16)
    x[0, 0, :] = 1.0  # token 0: all neurons = +1
    x[0, 1, :] = -1.0  # token 1: all neurons = -1

    _ = model(x)
    # Signed mean over seq would be 0 for every neuron → _running_mean_abs all zeros.
    # Abs-first mean should be 1.0 for every neuron.
    assert cb._running_mean_abs is not None
    assert cb._running_mean_abs.min().item() > 0.5
    cb.on_train_end(args, state, control)


def test_hook_excludes_padding_via_attention_mask():
    """Padding positions (mask=0) must not contribute to the novelty reduction.

    Two identical-content batches that differ only in padding length should
    produce the same running stats.
    """
    cb = _import_callback()(layer_getter = lambda m: m.mlp)
    model = _TinySeqModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)

    with torch.no_grad():
        model.mlp.fc.weight.copy_(torch.eye(16))
        model.mlp.fc.bias.zero_()

    # One real token followed by one padding token
    x_padded = torch.ones(1, 2, 16)
    x_padded[0, 1, :] = 999.0  # padding — should be ignored
    mask = torch.tensor([[1, 0]])  # only token 0 is real

    # Simulate the mask being cached as if the pre-forward hook ran
    cb._cached_mask = mask.float()
    _ = model(x_padded)

    # Stats should reflect only the real token (value ≈ 1.0), not the pad (999)
    assert cb._running_mean_abs is not None
    assert cb._running_mean_abs.max().item() < 10.0

    cb.on_train_end(args, state, control)


def test_auto_layer_detection():
    """_find_mlp_layers should locate the 'mlp' sub-module automatically."""
    cb = _import_callback()()  # no layer_getter
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model = model)
    assert cb._handle is not None
    cb.on_train_end(args, state, control)


# ---------------------------------------------------------------------------
# on_evaluate
# ---------------------------------------------------------------------------


def test_on_evaluate_resets_running_stats():
    cb = _import_callback()(layer_getter = lambda m: m.mlp)
    model = _TinyModel()
    args, state, control = _make_state_control()

    cb.on_train_begin(args, state, control, model = model)
    model.eval()
    _ = model(torch.randn(4, 16))
    assert cb._running_mean_abs is not None

    cb.on_evaluate(args, state, control, model = model)
    # Stats cleared after each eval so the next window starts fresh.
    assert cb._running_mean_abs is None
    assert cb._running_count == 0


def test_on_evaluate_updates_history():
    cb = _import_callback()(layer_getter = lambda m: m.mlp, window = 3)
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)

    for _ in range(2):
        model.eval()
        _ = model(torch.randn(4, 16))
        model.train()
        cb.on_evaluate(args, state, control, model = model)

    assert len(cb._history) == 2


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


def test_early_stop_triggers_after_window():
    cb = _import_callback()(
        layer_getter = lambda m: m.mlp,
        novelty_threshold = 0.5,
        window = 2,
        early_stop = True,
    )
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)

    # Inject spike running stats -> novelty << threshold
    for _ in range(2):
        x = torch.zeros(4, 16)
        x[:, 0] = 1.0
        _inject_mean_abs(cb, x)
        control = cb.on_evaluate(args, state, control, model = model)

    assert control.should_training_stop is True


def test_early_stop_not_triggered_before_window():
    cb = _import_callback()(
        layer_getter = lambda m: m.mlp,
        novelty_threshold = 0.5,
        window = 3,
        early_stop = True,
    )
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)

    # Only 2 evals, window=3 -> should not stop yet
    for _ in range(2):
        x = torch.zeros(4, 16)
        x[:, 0] = 1.0
        _inject_mean_abs(cb, x)
        control = cb.on_evaluate(args, state, control, model = model)

    assert not control.should_training_stop


def test_early_stop_false_never_stops():
    cb = _import_callback()(
        layer_getter = lambda m: m.mlp,
        novelty_threshold = 0.99,
        window = 1,
        early_stop = False,
    )
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)

    x = torch.zeros(4, 16)
    x[:, 0] = 1.0
    _inject_mean_abs(cb, x)
    control = cb.on_evaluate(args, state, control, model = model)

    assert not control.should_training_stop


def test_early_stop_does_not_trigger_on_high_novelty():
    cb = _import_callback()(
        layer_getter = lambda m: m.mlp,
        novelty_threshold = 0.05,
        window = 2,
        early_stop = True,
    )
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)

    # Uniform activations -> high novelty, should not stop
    for _ in range(3):
        _inject_mean_abs(cb, torch.ones(4, 16))
        control = cb.on_evaluate(args, state, control, model = model)

    assert not control.should_training_stop


# ---------------------------------------------------------------------------
# on_log
# ---------------------------------------------------------------------------


def test_on_log_injects_metric():
    cb = _import_callback()()
    cb._last_novelty = 0.77
    logs = {"loss": 1.5}
    args, state, control = _make_state_control()
    cb.on_log(args, state, control, logs = logs)
    assert logs["activation_novelty"] == pytest.approx(0.77)


def test_on_log_custom_key():
    cb = _import_callback()(log_key = "rep_novelty")
    cb._last_novelty = 0.33
    logs = {}
    args, state, control = _make_state_control()
    cb.on_log(args, state, control, logs = logs)
    assert "rep_novelty" in logs
    assert logs["rep_novelty"] == pytest.approx(0.33)


def test_on_log_no_crash_when_logs_none():
    cb = _import_callback()()
    args, state, control = _make_state_control()
    cb.on_log(args, state, control, logs = None)  # must not raise


def test_on_log_injects_fresh_novelty_when_eval_metrics_present():
    """on_log must compute current novelty (not stale _last_novelty) when the
    log row contains eval_ keys — matching the HF Trainer call order where
    on_log fires before on_evaluate.
    """
    cb = _import_callback()(layer_getter = lambda m: m.mlp)
    model = _TinyModel()
    args, state, control = _make_state_control()
    cb.on_train_begin(args, state, control, model = model)

    # Stale value left over from a previous eval
    cb._last_novelty = 0.99

    # Inject spike stats -> fresh novelty would be near 0
    x = torch.zeros(4, 16)
    x[:, 0] = 1.0
    _inject_mean_abs(cb, x)

    logs = {"eval_loss": 0.5, "eval_runtime": 1.0}
    cb.on_log(args, state, control, logs = logs)

    # Should inject fresh (low) novelty, not the stale 0.99
    assert logs[cb.log_key] < 0.05
    # Stats must be cleared so on_evaluate does not double-count
    assert cb._running_mean_abs is None
    assert cb._running_count == 0
