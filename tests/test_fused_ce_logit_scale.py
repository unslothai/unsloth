# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The fused CE branch must apply the same logit transforms as the eager branch.

Cohere carries ``logit_scale`` (logits are multiplied by it) and Granite carries
``logits_scaling`` (logits are divided by it). The eager branch reads both through
``detect_logit_transforms`` and hands them to ``fast_cross_entropy_loss``; the fused
branch, which is the default whenever labels are present, has to pass them to
``unsloth_fused_ce_loss`` or training optimizes a different loss than the reference
implementation computes. Pure torch, so it runs on CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import CohereConfig, FalconH1Config, GraniteConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

import pytest

from unsloth.models.llama import CausalLM_fast_forward


VOCAB_SIZE = 8
HIDDEN_SIZE = 4
BSZ = 2
Q_LEN = 4
LABELS = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])


def _hidden_states():
    """Deterministic without a seed, so the expected value survives a PRNG change."""
    values = torch.arange(BSZ * Q_LEN * HIDDEN_SIZE, dtype = torch.float32)
    return ((values % 7) - 3.0).reshape(BSZ, Q_LEN, HIDDEN_SIZE)


def _lm_head_weight():
    values = torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype = torch.float32)
    return ((values % 5) - 2.0).reshape(VOCAB_SIZE, HIDDEN_SIZE)


class _FakeModel:
    """Stands in for the decoder stack: the branch under test only reads outputs[0]."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states

    def __call__(self, **kwargs):
        return BaseModelOutputWithPast(
            last_hidden_state = self.hidden_states,
            past_key_values = None,
            hidden_states = None,
            attentions = None,
        )


class _FakeCausalLM:
    def __init__(self, config):
        self.config = config
        self.model = _FakeModel(_hidden_states())
        self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias = False)
        with torch.no_grad():
            self.lm_head.weight.copy_(_lm_head_weight())


def _fused_loss(config, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    forward = CausalLM_fast_forward(None)
    output = forward(_FakeCausalLM(config), labels = LABELS.clone(), return_dict = True)
    return output.loss.item()


def _reference_loss(scale):
    """Transformers' own order: scale the logits, shift, then cross entropy.

    Computed in float64 so the expected value is the arithmetic, not the float32 path.
    """
    logits = F.linear(_hidden_states().double(), _lm_head_weight().double())
    logits = logits * scale
    shifted = torch.empty_like(LABELS)
    shifted[..., :-1] = LABELS[..., 1:]
    shifted[..., -1] = -100
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), shifted.reshape(-1))
    return loss.item()


def _config(cls, **kwargs):
    return cls(
        hidden_size = HIDDEN_SIZE,
        intermediate_size = 2 * HIDDEN_SIZE,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        vocab_size = VOCAB_SIZE,
        **kwargs,
    )


@pytest.mark.parametrize(
    "config,scale",
    [
        # Cohere / Command-R / Aya multiply, Granite 3 divides.
        (_config(CohereConfig, logit_scale = 0.0625), 0.0625),
        (_config(GraniteConfig, logits_scaling = 4.0), 1.0 / 4.0),
    ],
    ids = ["cohere", "granite"],
)
def test_fused_ce_applies_the_configured_logit_scale(config, scale, monkeypatch):
    unscaled = _reference_loss(1.0)  # 7.898634, what the fused branch returns unfixed
    expected = _reference_loss(scale)
    assert abs(expected - unscaled) > 1.0, "the fixture no longer separates the two losses"

    # float32 accumulation over 8 logits, so well inside the default float32 tolerance.
    assert _fused_loss(config, monkeypatch) == pytest.approx(expected, rel = 1e-6), (
        f"fused CE dropped {config.model_type}'s logit scale: it returns the unscaled "
        f"{unscaled:.6f} instead of {expected:.6f}"
    )


def test_fused_ce_applies_falcon_h1_multiplier_once(monkeypatch):
    """The multiplier is folded into the hidden states, so it must not scale the logits too."""
    config = _config(FalconH1Config, lm_head_multiplier = 3.0)
    assert _fused_loss(config, monkeypatch) == pytest.approx(_reference_loss(3.0), rel = 1e-6)
