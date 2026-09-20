# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every loss branch must apply the logit transforms the reference implementation does.

Cohere multiplies by logit_scale, Granite divides by logits_scaling; dropping either
optimizes a different loss. Pure torch, so it runs on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import (
    CohereConfig,
    FalconH1Config,
    Gemma2Config,
    GraniteConfig,
    MistralConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

import pytest

import unsloth.models.llama as llama_module
import unsloth.models.mistral as mistral_module
from unsloth.models.llama import (
    apply_logit_transforms,
    CausalLM_fast_forward,
    resolve_logit_transforms,
)
from unsloth.models.mistral import MistralForCausalLM_fast_forward


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
    # mistral.py reads self.training before either loss branch; without this the tests
    # pass or error depending on which attention backend a previous test selected.
    training = False

    def __init__(self, config):
        self.config = config
        self.model = _FakeModel(_hidden_states())
        self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias = False)
        with torch.no_grad():
            self.lm_head.weight.copy_(_lm_head_weight())


def _fused_loss(
    config,
    monkeypatch,
    forward = None,
):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    extra = {}
    if forward is None:
        forward = CausalLM_fast_forward(None)
    else:
        # mistral.py reads input_ids.shape before it reaches the decoder stack.
        extra["input_ids"] = torch.zeros(BSZ, Q_LEN, dtype = torch.long)
    output = forward(_FakeCausalLM(config), labels = LABELS.clone(), return_dict = True, **extra)
    return output.loss.item()


def _reference_loss(scale, softcapping = 0.0):
    """Transformers' own order: scale the logits, soft cap, shift, then cross entropy.

    Computed in float64 so the expected value is the arithmetic, not the float32 path.
    """
    logits = F.linear(_hidden_states().double(), _lm_head_weight().double())
    logits = logits * scale
    if softcapping:
        logits = softcapping * torch.tanh(logits / softcapping)
    shifted = torch.empty_like(LABELS)
    shifted[..., :-1] = LABELS[..., 1:]
    shifted[..., -1] = -100
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), shifted.reshape(-1))
    return loss.item()


def _config(cls, **kwargs):
    """Deferred: transformers validates strictly and per release, so eager construction
    turns one unconstructable class into a collection error for the whole file."""

    def build():
        try:
            return cls(
                hidden_size = HIDDEN_SIZE,
                intermediate_size = 2 * HIDDEN_SIZE,
                num_hidden_layers = 1,
                num_attention_heads = 2,
                vocab_size = VOCAB_SIZE,
                **kwargs,
            )
        except Exception as error:
            pytest.skip(f"{cls.__name__} does not accept this shape here: {error}")

    build.cls = cls
    return build


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
    config = config()
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
    config = _config(FalconH1Config, lm_head_multiplier = 3.0)()
    assert _fused_loss(config, monkeypatch) == pytest.approx(_reference_loss(3.0), rel = 1e-6)


def test_mistral_fused_ce_reads_the_transforms_the_same_way(monkeypatch):
    """No shipped Mistral config carries a scale, so one is injected: the point is that
    the call site forwards what the config holds, not that Mistral needs it."""
    plain = _config(MistralConfig)()
    assert _fused_loss(plain, monkeypatch, MistralForCausalLM_fast_forward) == pytest.approx(
        _reference_loss(1.0),
        rel = 1e-6,
    ), "plain Mistral carries no transform and must be unaffected"

    scaled = _config(MistralConfig)()
    scaled.logit_scale = 0.0625
    assert _fused_loss(scaled, monkeypatch, MistralForCausalLM_fast_forward) == pytest.approx(
        _reference_loss(0.0625),
        rel = 1e-6,
    ), "mistral.py's fused branch dropped the configured logit scale"


@pytest.mark.parametrize(
    "config,expected",
    [
        (_config(CohereConfig, logit_scale = 0.0625), (0, 0.0625, 0)),
        (_config(GraniteConfig, logits_scaling = 4.0), (0, 0, 4.0)),
        # The exact model_type test the fallback used to do missed the MoE spellings.
        (_config(GraniteConfig, logits_scaling = 4.0, model_type = "granitemoe"), (0, 0, 4.0)),
        (_config(FalconH1Config, lm_head_multiplier = 3.0), (0, 3.0, 0)),
    ],
    ids = ["cohere", "granite", "granitemoe", "falcon_h1"],
)
def test_transforms_resolve_without_unsloth_zoo(config, expected, monkeypatch):
    config = config()
    """The fallback arm runs whenever unsloth_zoo predates detect_logit_transforms."""
    monkeypatch.setattr(llama_module, "detect_logit_transforms", None)
    # Both arms agree on these configs, so without this the coverage could evaporate
    # silently if the patch stopped reaching the code under test.
    assert llama_module.detect_logit_transforms is None
    assert resolve_logit_transforms(config) == pytest.approx(expected)


@pytest.mark.parametrize(
    "model_type,field",
    [
        ("cohere", "logit_scale"),
        ("granite", "logits_scaling"),
        ("falcon_h1", "lm_head_multiplier"),
        ("gemma2", "final_logit_softcapping"),
    ],
)
def test_a_none_valued_field_resolves_to_zero(model_type, field, monkeypatch):
    """None must read as "off". A namespace, not a config: transformers 5 rejects None at
    construction, but a checkpoint carrying one still loads into an older config."""
    monkeypatch.setattr(llama_module, "detect_logit_transforms", None)
    config = SimpleNamespace(model_type = model_type, **{field: None})
    assert resolve_logit_transforms(config) == (0, 0, 0)


def test_a_none_model_type_does_not_raise(monkeypatch):
    """Remote-code configs do set model_type to None, and `in`-style reads must survive it."""
    monkeypatch.setattr(llama_module, "detect_logit_transforms", None)
    config = _config(CohereConfig, logit_scale = 0.0625)()
    config.model_type = None
    assert resolve_logit_transforms(config) == (0, 0.0625, 0)


def _inference_logits(model, forward):
    extra = {"input_ids": torch.zeros(BSZ, Q_LEN, dtype = torch.long)} if forward else {}
    forward = forward or CausalLM_fast_forward(None)
    return forward(model, labels = None, return_dict = True, **extra).logits


_SCALE_CASES = [
    (_config(CohereConfig, logit_scale = 0.0625), 0.0625, 0.0),
    # Granite exercises the divisor fold, which the fused branch never takes.
    (_config(GraniteConfig, logits_scaling = 4.0), 1.0 / 4.0, 0.0),
    (_config(Gemma2Config, final_logit_softcapping = 4.0), 1.0, 4.0),
]
_SCALE_IDS = ["cohere", "granite", "gemma2_softcap"]


@pytest.mark.parametrize(
    "forward", [None, MistralForCausalLM_fast_forward], ids = ["llama", "mistral"]
)
@pytest.mark.parametrize("config,scale,softcapping", _SCALE_CASES, ids = _SCALE_IDS)
def test_inference_logits_carry_the_same_transforms(config, scale, softcapping, forward):
    config = config()
    """The labels-free branch returns the logits, so they must arrive already transformed."""
    expected = F.linear(_hidden_states(), _lm_head_weight()) * scale
    if softcapping:
        expected = softcapping * torch.tanh(expected / softcapping)
    model = _FakeCausalLM(config)
    assert torch.allclose(_inference_logits(model, forward), expected, atol = 1e-5)
    # Applied in place, so a second call on the SAME model must not compound them.
    assert torch.allclose(_inference_logits(model, forward), expected, atol = 1e-5)


@pytest.mark.parametrize(
    "module,forward",
    [(llama_module, None), (mistral_module, MistralForCausalLM_fast_forward)],
    ids = ["llama", "mistral"],
)
@pytest.mark.parametrize("config,scale,softcapping", _SCALE_CASES, ids = _SCALE_IDS)
def test_materialized_branch_is_handed_the_same_transforms(
    config, scale, softcapping, module, forward, monkeypatch
):
    """The eager branch needs Triton, so assert on what it hands the kernel. Each module
    has its own binding via `from .llama import *`, so the spy goes on the call site's."""
    config = config()
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return torch.zeros((), requires_grad = True)

    monkeypatch.setattr(module, "fast_cross_entropy_loss", spy)
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    extra = {"input_ids": torch.zeros(BSZ, Q_LEN, dtype = torch.long)} if forward else {}
    (forward or CausalLM_fast_forward(None))(
        _FakeCausalLM(config),
        labels = LABELS.clone(),
        return_dict = True,
        **extra,
    )

    assert len(calls) == 1, "the eager branch did not run"
    assert calls[0]["logit_scaling"] == pytest.approx(scale if scale != 1.0 else 0)
    assert calls[0]["logit_softcapping"] == pytest.approx(softcapping)


def test_transforms_are_applied_scale_first_then_soft_cap():
    """tanh does not commute with scaling, and no shipped family does both, so nothing
    above separates the two orders."""
    logits = F.linear(_hidden_states(), _lm_head_weight())
    scale, softcapping = 0.0625, 4.0
    scale_first = softcapping * torch.tanh((logits * scale) / softcapping)
    cap_first = softcapping * torch.tanh(logits / softcapping) * scale
    assert not torch.allclose(scale_first, cap_first), "the fixture no longer separates the orders"
    assert torch.allclose(
        apply_logit_transforms(logits.clone(), softcapping, scale),
        scale_first,
        atol = 1e-6,
    )
