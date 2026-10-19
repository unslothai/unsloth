# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""GRPO `sampling/*` metrics must reduce over the loss mask, as TRL's _generate_and_score_completions does."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

RL_REPLACEMENTS = (
    Path(importlib.util.find_spec("unsloth").origin).parent / "models" / "rl_replacements.py"
)


def _helper():
    tree = ast.parse(RL_REPLACEMENTS.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_grpo_is_metric_values"
    ]
    assert len(nodes) == 1, "no _unsloth_grpo_is_metric_values in rl_replacements.py"
    ns = {"torch": torch}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(RL_REPLACEMENTS), "exec"), ns)
    return ns["_unsloth_grpo_is_metric_values"]


def _batch(sequence_level):
    torch.manual_seed(0)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype = torch.float32
    )
    old = torch.randn(4, 5) * 0.1 - 2.0
    sampling = old + torch.randn(4, 5) * 0.05
    log_ratio = (old - sampling) * mask
    if sequence_level:
        ratio = torch.exp(log_ratio.sum(-1, keepdim = True))
    else:
        ratio = torch.exp(log_ratio)
    return mask, old, sampling, ratio


def _trl_reference(mask, old, sampling, ratio, sequence_level):
    # TRL 1.15 grpo_trainer.py `if self.use_vllm and self.vllm_importance_sampling_correction:` block.
    keep = mask.bool()
    delta = torch.abs(old - sampling)
    delta = delta[keep & ~torch.isnan(delta)]
    flat = ratio.flatten() if sequence_level else ratio[keep]
    return delta, flat


def _unsloth_inputs(mask, old, sampling, ratio):
    # What grpo_compute_loss returns: both zero-filled outside the mask, ratio broadcast to (B, T).
    return torch.abs(old - sampling) * mask, ratio * mask


@pytest.mark.parametrize("sequence_level", [False, True])
def test_metric_values_match_trl(sequence_level):
    mask, old, sampling, ratio = _batch(sequence_level)
    delta, flat = _unsloth_inputs(mask, old, sampling, ratio)
    got_delta, got_ratio = _helper()(delta, flat, mask, sequence_level)
    want_delta, want_ratio = _trl_reference(mask, old, sampling, ratio, sequence_level)
    for reduce in (torch.mean, torch.min, torch.max):
        assert torch.allclose(reduce(got_delta), reduce(want_delta))
        assert torch.allclose(reduce(got_ratio), reduce(want_ratio))


def test_unfiltered_values_read_zero_min_ratio():
    # Control: reducing the zero-filled tensors directly is what the metrics did before.
    mask, old, sampling, ratio = _batch(False)
    delta, flat = _unsloth_inputs(mask, old, sampling, ratio)
    assert flat.min().item() == 0.0
    assert _helper()(delta, flat, mask, False)[1].min().item() > 0.5


def test_shape_mismatch_keeps_the_old_reduction():
    mask, old, sampling, ratio = _batch(False)
    delta, flat = _unsloth_inputs(mask, old, sampling, ratio)
    got_delta, got_ratio = _helper()(delta, flat, mask[:, :3], False)
    assert got_delta.numel() == delta.numel() and got_ratio.numel() == flat.numel()
