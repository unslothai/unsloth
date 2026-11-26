# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LoRAs on an NVFP4 transformer bake at load like int8 / fp8: the adapters attach to the dense
model, only the base Linears are quantized, and a per-layer policy sees the plain model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from core.inference.diffusion_lora import (  # noqa: E402
    _DIFFUSERS_LORA_BAKED_QUANT,
    _DIFFUSERS_LORA_BLOCKED_QUANT,
    lora_unavailable_reason,
    supports_lora,
)
from core.inference.diffusion_nvfp4_policy import (  # noqa: E402
    NVFP4Policy,
    PolicyMismatch,
    Rule,
    assign_precisions,
)


def _gate(quant):
    return dict(
        engine = "diffusers", family = "flux.2-klein", model_kind = "diffusers", transformer_quant = quant
    )


@pytest.mark.parametrize("quant", ["nvfp4", "NVFP4"])
def test_nvfp4_transformer_takes_loras(quant):
    assert lora_unavailable_reason(**_gate(quant)) is None
    assert supports_lora(**_gate(quant))


def test_nvfp4_bakes_even_when_compiled():
    # A baked build carries its adapters inside the compiled quantized graph, as int8 / fp8 do.
    assert lora_unavailable_reason(compiled = True, **_gate("nvfp4")) is None


def test_nvfp4_is_baked_and_mxfp8_still_blocked():
    assert "nvfp4" in _DIFFUSERS_LORA_BAKED_QUANT
    assert "nvfp4" not in _DIFFUSERS_LORA_BLOCKED_QUANT
    assert "mxfp8" in _DIFFUSERS_LORA_BLOCKED_QUANT
    assert "MXFP8" in (lora_unavailable_reason(**_gate("mxfp8")) or "")


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(width, width))


class _Tiny(nn.Module):
    def __init__(
        self,
        width: int = 64,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width) for _ in range(blocks)])
        self.proj_out = nn.Linear(width, 8)


_POLICY = NVFP4Policy(
    policy_id = "test_lora_v1",
    version = 1,
    family = "test",
    base_repos = ("test/tiny",),
    rules = (Rule(suffix = "mod.1", precision = "nvfp4", expect = 2),),
    expected_counts = {"nvfp4": 2, "fp8": 4, "bf16": 1},
)


def _model() -> nn.Module:
    torch.manual_seed(0)
    return _Tiny().to(torch.bfloat16)


def _with_lora(model: nn.Module) -> nn.Module:
    from peft import LoraConfig, inject_adapter_in_model
    config = LoraConfig(r = 4, lora_alpha = 4, target_modules = ["to_q", "to_k", "mod.1"])
    return inject_adapter_in_model(config, model, adapter_name = "style")


def test_policy_assignment_is_unchanged_by_attached_adapters():
    plain = assign_precisions(_model(), _POLICY, min_features = 16)
    model = _with_lora(_model())
    assert any(".lora_A." in n for n, _ in model.named_modules())
    baked = assign_precisions(model, _POLICY, min_features = 16)
    # Same precisions on the same layers, addressed through the adapter's base_layer.
    unwrapped = {k.replace(".base_layer", ""): v for k, v in baked.items()}
    assert unwrapped == plain
    assert all(".lora_" not in k for k in baked)
    assert baked["blocks.0.mod.1.base_layer"] == "nvfp4"
    assert baked["blocks.1.to_q.base_layer"] == "fp8"


def test_policy_still_refuses_a_model_it_was_not_written_for():
    model = _with_lora(_Tiny(blocks = 3).to(torch.bfloat16))
    with pytest.raises(PolicyMismatch):
        assign_precisions(model, _POLICY, min_features = 16)
