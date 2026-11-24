# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""save_pretrained of a model whose fused expert stacks zoo quantized in place (bare Params4bit) under a model
converter (Inkling 2: NVFP4 dequantize + Interleave of `w13_weight` into `gate_up_proj`). transformers reverses
that converter on save, on the packed bytes and on the QuantState entries beside them, which raises or writes
garbage. The patched revert skips it: the stack is written in the model's layout under the model's own name (the
renames still reverted), which zoo's pre-quantized expert loaders read back. A dense stack keeps its converter."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
bnb = pytest.importorskip("bitsandbytes")

from unsloth.import_fixes import (  # noqa: E402
    _BNB_PREQUANTIZED_SAVE_FLAG,
    fix_transformers_bnb_prequantized_save,
)


def _modules():
    try:
        from transformers import core_model_loading
    except Exception:
        pytest.skip("this transformers has no core_model_loading")
    for name in ("WeightConverter", "WeightRenaming", "Interleave", "revert_weight_conversion"):
        if not hasattr(core_model_loading, name):
            pytest.skip(f"this transformers has no core_model_loading.{name}")
    return core_model_loading


def _unwrapped(fn):
    while getattr(fn, _BNB_PREQUANTIZED_SAVE_FLAG, False):
        fn = fn.__wrapped__
    return fn


def _params4bit(value, module):
    """A 3-D expert stack as zoo's Bnb4bitQuantize patch leaves it (nf4, double quantized, packed [n, 1])."""
    data, state = bnb.functional.quantize_4bit(
        value.contiguous(), blocksize = 64, compress_statistics = True, quant_type = "nf4"
    )
    new = torch.Tensor._make_subclass(bnb.nn.Params4bit, data, False)
    new.quant_state = state
    new.blocksize = 64
    new.compress_statistics = True
    new.quant_type = "nf4"
    new.quant_storage = torch.uint8
    new.bnb_quantized = True
    new.module = module
    new._original_shape = value.shape
    return new


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(4, 64, 32), requires_grad = False)


class _Model(torch.nn.Module):
    def __init__(self, quantized):
        super().__init__()
        torch.manual_seed(0)
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.experts = _Experts()
        self.model.language_model.norm = torch.nn.LayerNorm(32)
        self.config = None
        experts = self.model.language_model.experts
        if quantized:
            experts._parameters["gate_up_proj"] = _params4bit(experts.gate_up_proj.data, experts)


def _state_dict(model):
    out = {}
    for name, param in model.named_parameters():
        out[name] = param.data.clone()
        state = getattr(param, "quant_state", None)
        if state is not None:
            for key, value in state.as_dict(packed = True).items():
                out[f"{name}.{key}"] = value.clone()
    return out


def _conversions(cml):
    return [
        cml.WeightRenaming(
            source_patterns = r"model\.llm\.", target_patterns = "model.language_model."
        ),
        cml.WeightConverter(
            source_patterns = r"experts\.w13_weight",
            target_patterns = "experts.gate_up_proj",
            operations = [cml.Interleave(dim = 1)],
        ),
    ]


def _saved_name(name):
    return name.replace("model.language_model.", "model.llm.", 1)


def test_upstream_revert_cannot_save_a_packed_expert_stack():
    cml = _modules()
    model = _Model(quantized = True)
    model._weight_conversions = _conversions(cml)
    state_dict = _state_dict(model)
    original = _unwrapped(cml.revert_weight_conversion)
    try:
        saved = original(model, dict(state_dict))
    except Exception:
        return  # transformers 5.24: Interleave reshapes the packed [n, 1] bytes and raises
    key = "model.language_model.experts.gate_up_proj"
    kept = {_saved_name(k): v for k, v in state_dict.items() if k.startswith(key)}
    assert not all(k in saved and torch.equal(saved[k], v) for k, v in kept.items())


def test_patched_revert_writes_the_packed_stack_in_model_layout():
    cml = _modules()
    fix_transformers_bnb_prequantized_save()
    patched = cml.revert_weight_conversion
    assert getattr(patched, _BNB_PREQUANTIZED_SAVE_FLAG, False)

    model = _Model(quantized = True)
    conversions = _conversions(cml)
    model._weight_conversions = conversions
    state_dict = _state_dict(model)
    saved = patched(model, dict(state_dict))
    assert model._weight_conversions is conversions

    assert set(saved) == {_saved_name(k) for k in state_dict}
    for key, value in state_dict.items():
        assert torch.equal(saved[_saved_name(key)], value), key
    stack = "model.llm.experts.gate_up_proj"
    assert {
        f"{stack}.absmax",
        f"{stack}.quant_map",
        f"{stack}.quant_state.bitsandbytes__nf4",
    } <= set(saved)

    # The saved entries rebuild bitwise the same 4-bit stack.
    param = model.model.language_model.experts.gate_up_proj
    stats = {k[len(stack) + 1 :]: v for k, v in saved.items() if k.startswith(stack + ".")}
    state = bnb.functional.QuantState.from_dict(stats, device = "cpu")
    assert tuple(state.shape) == (4, 64, 32)
    want = bnb.functional.dequantize_4bit(param.data, param.quant_state)
    got = bnb.functional.dequantize_4bit(saved[stack], state)
    assert torch.equal(want, got)


def test_patched_revert_keeps_the_converter_of_a_dense_stack():
    cml = _modules()
    fix_transformers_bnb_prequantized_save()
    patched = cml.revert_weight_conversion
    model = _Model(quantized = False)
    model._weight_conversions = _conversions(cml)
    state_dict = _state_dict(model)
    want = _unwrapped(patched)(model, dict(state_dict))
    got = patched(model, dict(state_dict))
    assert set(got) == set(want)
    assert "model.llm.experts.w13_weight" in got
    for key in want:
        assert torch.equal(got[key], want[key])
