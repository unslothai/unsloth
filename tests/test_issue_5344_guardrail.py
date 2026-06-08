"""Unit tests for the unslothai/unsloth#5344 silent-quantization-bypass guardrail.

Covers two failure modes the helper detects:
  1. total bypass: load_in_4bit was requested but no bnb modules exist.
  2. partial bypass: bnb quantized nn.Linear but a large fraction of weight
     bytes live in non-nn.Linear Parameters (e.g. Gemma-4 MoE fused experts).
"""

import warnings

import torch
import torch.nn as nn


# unsloth must be imported before transformers per its loading order, but
# these tests do not exercise the real loader. Import the helper directly.
from unsloth.models.vision import _warn_if_quantization_silently_dropped


class _PretendLinear4bit(nn.Module):
    """type(m).__name__ == 'Linear4bit' so the guardrail counts it as quantized."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1, dtype = torch.uint8),
            requires_grad = False,
        )


_PretendLinear4bit.__name__ = "Linear4bit"


def _unquantized_model():
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))


def _quantized_model():
    return nn.Sequential(nn.Linear(4, 4), _PretendLinear4bit())


def test_fires_when_4bit_requested_but_no_bnb_modules():
    model = _unquantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("load_in_4bit=True was requested" in m for m in msgs), msgs
    assert any("issues/5344" in m for m in msgs), msgs


def test_silent_when_4bit_succeeded():
    model = _quantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("load_in_4bit" in m for m in msgs), msgs


def test_silent_for_full_finetuning():
    model = _unquantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = False,
            full_finetuning = True,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("load_in_4bit" in m or "load_in_8bit" in m for m in msgs), msgs


def test_silent_when_no_quantization_requested():
    model = _unquantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("load_in_4bit" in m or "load_in_8bit" in m for m in msgs), msgs


def test_fires_for_8bit_silent_bypass():
    model = _unquantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = True,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("load_in_8bit=True was requested" in m for m in msgs), msgs


class _MoEFusedExpertWrapper(nn.Module):
    """Mimics Gemma4TextExperts: fused 3D weights stored as nn.Parameter, not
    as separate nn.Linear instances. bnb's replace_with_bnb_linear skips this."""

    def __init__(self, num_experts = 128, intermediate = 1408, hidden = 2816):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.zeros((num_experts, intermediate, hidden), dtype = torch.bfloat16),
            requires_grad = False,
        )


def _partial_quant_model():
    return nn.Sequential(_PretendLinear4bit(), _MoEFusedExpertWrapper())


def test_fires_on_partial_quant_moe_experts():
    model = _partial_quant_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs
    assert any("gate_up_proj" in m for m in msgs), msgs


class _NormParam(nn.Module):
    """An RMSNorm-like module: large BF16 weight whose name is in the skip list."""

    def __init__(self, dim = 8 * 1024 * 1024 + 10):
        super().__init__()
        self.norm_weight = nn.Parameter(
            torch.zeros(dim, dtype = torch.bfloat16),
            requires_grad = False,
        )


def test_silent_when_only_skip_list_tensors_unquantized():
    model = nn.Sequential(_PretendLinear4bit(), _NormParam())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs


if __name__ == "__main__":
    test_fires_when_4bit_requested_but_no_bnb_modules()
    test_silent_when_4bit_succeeded()
    test_silent_for_full_finetuning()
    test_silent_when_no_quantization_requested()
    test_fires_for_8bit_silent_bypass()
    test_fires_on_partial_quant_moe_experts()
    test_silent_when_only_skip_list_tensors_unquantized()
    print("All 7 guardrail tests passed.")
