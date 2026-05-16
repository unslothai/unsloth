# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-expert bitsandbytes Linear4bit swap for Gemma-4 MoE experts.

Refs: https://github.com/unslothai/unsloth/issues/5344

Gemma4TextExperts stores all experts as two fused 3D Parameters
(gate_up_proj, down_proj) shaped (num_experts, out_dim, in_dim) so that
torch._grouped_mm can dispatch a single grouped matmul per layer. The
fused storage is great for forward throughput but breaks bnb 4-bit
quantization: bnb.nn.Linear4bit only swaps nn.Linear instances, so the
fused 3D Parameters stay in BF16, defeating QLoRA VRAM savings.

This module swaps each Gemma4TextExperts module's fused weights for two
nn.ModuleList[Linear4bit] of length num_experts, and overrides forward to
dispatch per-expert. The trade-off is the loss of torch._grouped_mm
throughput in exchange for a ~4x reduction in expert weight VRAM
(measured on unsloth/gemma-4-26B-A4B-it: 46 GB -> 14.27 GB resident).

Gated on UNSLOTH_GEMMA4_MOE_4BIT (default off) and on load_in_4bit=True.
Default off until the matching per-expert LoRA path lands; opt in via
the env var if you want the VRAM win without QLoRA training.
"""

from __future__ import annotations

import os
from types import MethodType

import torch
import torch.nn as nn


__all__ = [
    "is_gemma4_moe_4bit_enabled",
    "swap_gemma4_experts_to_per_expert_linear4bit",
]


def is_gemma4_moe_4bit_enabled() -> bool:
    """Opt-in via UNSLOTH_GEMMA4_MOE_4BIT=1."""
    return os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT", "0") == "1"


def _per_expert_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Replacement Gemma4TextExperts.forward dispatching through swapped
    nn.ModuleList[Linear4bit] instead of fused 3D Parameters."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index,
            num_classes = self.num_experts,
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim = (-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = self.gate_up_proj_4bit[expert_idx](current_state)
        gate, up = gate_up.chunk(2, dim = -1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = self.down_proj_4bit[expert_idx](current_hidden_states)
        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0,
            token_idx,
            current_hidden_states.to(final_hidden_states.dtype),
        )

    return final_hidden_states


def _quantize_one_expert_to_linear4bit(
    weight_2d: torch.Tensor,
    compute_dtype: torch.dtype,
    quant_type: str = "nf4",
):
    """Build a bnb.nn.Linear4bit from a single (out, in) weight slice.
    Params4bit triggers on-the-fly quantization on .to(device)."""
    import bitsandbytes as bnb

    out_features, in_features = weight_2d.shape
    layer = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias = False,
        compute_dtype = compute_dtype,
        quant_type = quant_type,
        quant_storage = torch.uint8,
    )
    layer.weight = bnb.nn.Params4bit(
        data = weight_2d.detach().clone().contiguous(),
        requires_grad = False,
        quant_type = quant_type,
    )
    return layer


def swap_gemma4_experts_to_per_expert_linear4bit(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    verbose: bool = False,
) -> int:
    """Find every Gemma4TextExperts module in `model`, replace its fused 3D
    weights with two nn.ModuleList[Linear4bit] (per-expert), and patch
    forward to dispatch per-expert.

    Returns the count of swapped modules. Zero if the model has no Gemma-4
    MoE experts or if transformers does not expose Gemma4TextExperts.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    except Exception:
        return 0

    swapped = 0
    for module in model.modules():
        if not isinstance(module, Gemma4TextExperts):
            continue
        # Idempotent: once swapped, the fused 3D Parameters are gone.
        if hasattr(module, "_unsloth_gemma4_moe_4bit_swapped"):
            continue
        if not hasattr(module, "gate_up_proj") or not hasattr(module, "down_proj"):
            continue

        gate_up = module.gate_up_proj
        down = module.down_proj
        if not isinstance(gate_up, nn.Parameter) or gate_up.ndim != 3:
            continue
        if not isinstance(down, nn.Parameter) or down.ndim != 3:
            continue

        num_experts, two_intermediate, hidden = gate_up.shape
        num_experts_d, hidden_d, intermediate = down.shape
        if (
            num_experts != num_experts_d
            or hidden != hidden_d
            or two_intermediate != 2 * intermediate
        ):
            # Unrecognised layout: skip rather than risk corrupting weights.
            if verbose:
                print(
                    f"Unsloth: skipping Gemma4TextExperts swap due to "
                    f"unexpected shapes gate_up={tuple(gate_up.shape)} "
                    f"down={tuple(down.shape)}"
                )
            continue

        device = gate_up.device

        gate_up_list = nn.ModuleList()
        down_list = nn.ModuleList()
        for e in range(num_experts):
            gu = _quantize_one_expert_to_linear4bit(
                gate_up.data[e],
                compute_dtype = compute_dtype,
                quant_type = quant_type,
            )
            dp = _quantize_one_expert_to_linear4bit(
                down.data[e],
                compute_dtype = compute_dtype,
                quant_type = quant_type,
            )
            gate_up_list.append(gu.to(device))
            down_list.append(dp.to(device))

        # Per-module peak = fused BF16 + accumulated per-expert nf4; released here.
        del module.gate_up_proj
        del module.down_proj

        module.gate_up_proj_4bit = gate_up_list
        module.down_proj_4bit = down_list

        # Per-instance bind so sibling Gemma4TextExperts keep the class method.
        module.forward = MethodType(_per_expert_forward, module)
        module._unsloth_gemma4_moe_4bit_swapped = True

        swapped += 1

    if swapped > 0 and torch.cuda.is_available():
        # Free the cached fused tensors so post-swap VRAM reflects 4-bit.
        torch.cuda.empty_cache()

    return swapped
