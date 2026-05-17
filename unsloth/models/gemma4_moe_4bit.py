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
    "is_gemma4_moe_4bit_grouped_enabled",
    "is_gemma4_moe_4bit_grouped_active_only_enabled",
    "swap_gemma4_experts_to_per_expert_linear4bit",
]


def is_gemma4_moe_4bit_enabled() -> bool:
    """Opt-in via UNSLOTH_GEMMA4_MOE_4BIT=1."""
    return os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT", "0") == "1"


def is_gemma4_moe_4bit_grouped_enabled() -> bool:
    """Opt-in via UNSLOTH_GEMMA4_MOE_4BIT_GROUPED=1. Requires the base swap to
    also be enabled. Uses dequant-then-torch._grouped_mm per forward instead of
    a per-expert Linear4bit loop; trades transient BF16 staging buffer for
    grouped-GEMM throughput. See unslothai/unsloth#5344 follow-up."""
    return os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT_GROUPED", "0") == "1"


def is_gemma4_moe_4bit_grouped_active_only_enabled() -> bool:
    """Opt-in via UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_ACTIVE_ONLY=1. Requires both
    base swap + grouped. Dequantizes only the experts touched by the current
    batch's top-k routing (~k*B active per layer) instead of all num_experts.
    For Gemma-4 MoE (128 experts, top-k=4) this cuts dequant work an order of
    magnitude in autoregressive decode."""
    return os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_ACTIVE_ONLY", "0") == "1"


def is_gemma4_moe_4bit_grouped_cached_enabled() -> bool:
    """Opt-in via UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_CACHE=1. Layered on active-only:
    cache the BF16 dequantized expert weights in a per-module LRU. Decode
    revisits to the same expert skip the dequant entirely. Cache cap via
    UNSLOTH_GEMMA4_MOE_4BIT_CACHE_SIZE (default 8)."""
    return os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_CACHE", "0") == "1"


def is_gemma4_moe_4bit_grouped_pt_dequant_enabled() -> bool:
    """Opt-in via UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_PT_DEQUANT=1. Layered on
    grouped+active_only. Replaces bnb.functional.dequantize_4bit with a
    pure-tensor NF4 dequant so torch.compile can fuse the unpack + codebook
    lookup + stack + grouped_mm into one Inductor graph. Per-block absmax is
    pre-dequantized at swap time and cached on each Linear4bit, so the
    per-forward path is bnb-free."""
    return os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_PT_DEQUANT", "0") == "1"


def is_gemma4_moe_4bit_grouped_static_bf16_enabled() -> bool:
    """Opt-in via UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_STATIC_BF16=1. Dequant every
    expert ONCE on the first forward and keep the fused (E, 2I, H) / (E, H, I)
    BF16 tensors live for the lifetime of the module. Subsequent forwards skip
    dequant entirely. Wins back grouped_mm throughput at the cost of holding
    a permanent BF16 mirror -- i.e. peak VRAM rises back toward the BF16
    baseline. Useful as a speed-ceiling experiment and for inference workloads
    that have spare VRAM."""
    return os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_STATIC_BF16", "0") == "1"


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


# NF4 codebook (16 entries, bnb convention). Used by the pure-PyTorch dequant
# path which torch.compile can fuse with the surrounding stack + grouped_mm.
_NF4_CODES = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype = torch.float32,
)

_COMPILED_PT_DEQUANT_STACK = None
_COMPILED_DEQUANT_STACK = None


def _ensure_pt_dequant_state(layer):
    """Cache the dequantized absmax + codebook on the layer so the per-forward
    path needs no bnb calls. Idempotent. Called at swap time."""
    if getattr(layer, "_unsloth_pt_dequant_ready", False):
        return
    from bitsandbytes.functional import dequantize_blockwise

    qs = layer.weight.quant_state
    if qs.nested:
        absmax_fp32 = dequantize_blockwise(qs.absmax, qs.state2)
        absmax_fp32 = (absmax_fp32 + qs.offset).to(torch.float32)
    else:
        absmax_fp32 = qs.absmax.to(torch.float32)
    layer._unsloth_pt_absmax_fp32 = absmax_fp32.contiguous()
    layer._unsloth_pt_blocksize = qs.blocksize
    layer._unsloth_pt_shape = tuple(qs.shape)
    layer._unsloth_pt_dtype = qs.dtype
    layer._unsloth_pt_dequant_ready = True


def _pt_dequant_one(packed_uint8, absmax_fp32, blocksize, shape, dtype, codes):
    """Pure-PyTorch NF4 dequant of one expert weight. Bit-exact vs bnb.
    Pure tensor ops -> torch.compile-friendly."""
    packed = packed_uint8.reshape(-1)
    high = (packed >> 4) & 0xF
    low = packed & 0xF
    indices = torch.stack([high, low], dim = -1).reshape(-1).to(torch.long)
    values = codes[indices]  # fp32
    n_elements = values.numel()
    n_blocks = (n_elements + blocksize - 1) // blocksize
    values = values.view(n_blocks, blocksize) * absmax_fp32.view(-1, 1)
    target = shape[0] * shape[1]
    return values.reshape(-1)[:target].view(shape).to(dtype)


def _pt_dequant_stack_subset(layers, indices_cpu, codes):
    """Pure-PyTorch dequant of a subset of experts and stack into (E_active, out, in)."""
    return torch.stack(
        [
            _pt_dequant_one(
                layers[i].weight.data,
                layers[i]._unsloth_pt_absmax_fp32,
                layers[i]._unsloth_pt_blocksize,
                layers[i]._unsloth_pt_shape,
                layers[i]._unsloth_pt_dtype,
                codes,
            )
            for i in indices_cpu
        ],
        dim = 0,
    )


def _get_compiled_pt_dequant_stack():
    """Lazy-compile the pure-PT dequant+stack helper. Re-used across forwards."""
    global _COMPILED_PT_DEQUANT_STACK
    if _COMPILED_PT_DEQUANT_STACK is None:
        _COMPILED_PT_DEQUANT_STACK = torch.compile(
            _pt_dequant_stack_subset,
            dynamic = True,
            fullgraph = False,
        )
    return _COMPILED_PT_DEQUANT_STACK


def _dequant_stack(layers):
    """Dequantize each Linear4bit in a ModuleList and stack into (E, out, in)."""
    from bitsandbytes.functional import dequantize_4bit

    return torch.stack(
        [dequantize_4bit(L.weight.data, L.weight.quant_state) for L in layers],
        dim = 0,
    )


def _dequant_stack_subset(layers, indices_cpu):
    """Dequantize only experts whose CPU-int indices are in indices_cpu."""
    from bitsandbytes.functional import dequantize_4bit

    return torch.stack(
        [
            dequantize_4bit(layers[i].weight.data, layers[i].weight.quant_state)
            for i in indices_cpu
        ],
        dim = 0,
    )


def _get_compiled_dequant_stack():
    """Lazily compile the dequant+stack helper. Done once and cached so
    successive forwards reuse the compiled graph (otherwise the per-call
    compile cost dwarfs the runtime saving)."""
    global _COMPILED_DEQUANT_STACK
    if _COMPILED_DEQUANT_STACK is None:
        _COMPILED_DEQUANT_STACK = torch.compile(
            _dequant_stack,
            dynamic = False,
            fullgraph = False,
        )
    return _COMPILED_DEQUANT_STACK


def _grouped_mm_forward_4bit(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Dequantize per-expert Linear4bit weights into a fused 3D BF16 tensor and
    run unsloth_zoo.forward_native_grouped_mm. Buffer is freed at end of each
    forward so resident VRAM stays at 4-bit. Transient peak per layer is
    (E * 2I * H + E * H * I) * 2 bytes BF16. Opt-in compile of the dequant
    helper via UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_COMPILE=1; bnb's CUDA dequant
    is a graph break so the win is bounded by kernel-launch overhead saved
    via CUDA-graph capture, not by Inductor fusion."""
    from unsloth_zoo.temporary_patches.moe_utils import (
        forward_native_grouped_mm,
    )

    if os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT_GROUPED_COMPILE", "0") == "1":
        dequant_stack = _get_compiled_dequant_stack()
    else:
        dequant_stack = _dequant_stack

    gate_up = dequant_stack(self.gate_up_proj_4bit)
    down = dequant_stack(self.down_proj_4bit)
    self.gate_up_proj = nn.Parameter(gate_up, requires_grad = False)
    self.down_proj = nn.Parameter(down, requires_grad = False)
    try:
        return forward_native_grouped_mm(
            self,
            hidden_states,
            top_k_index,
            top_k_weights,
        )
    finally:
        del self.gate_up_proj
        del self.down_proj


def _grouped_mm_forward_4bit_active_only(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Like _grouped_mm_forward_4bit but only dequants the experts that this
    batch's top-k routing actually touches. Remaps top_k_index from the full
    expert space [0, num_experts) to the compact active range [0, E_active)
    so torch._grouped_mm sees only the populated groups.

    For decode (S*K << num_experts) the saving is large; for prefill where
    most experts get hit the active set approaches num_experts and the
    overhead of the unique+remap is dominated by the dequant savings."""
    from unsloth_zoo.temporary_patches.moe_utils import (
        forward_native_grouped_mm,
    )

    flat = top_k_index.reshape(-1)
    active_experts, inverse = torch.unique(flat, return_inverse = True)
    active_cpu = active_experts.tolist()
    n_active = len(active_cpu)

    gate_up = _dequant_stack_subset(self.gate_up_proj_4bit, active_cpu)
    down = _dequant_stack_subset(self.down_proj_4bit, active_cpu)

    compact_top_k = inverse.view_as(top_k_index)

    saved_n_experts = self.num_experts
    self.num_experts = n_active
    self.gate_up_proj = nn.Parameter(gate_up, requires_grad = False)
    self.down_proj = nn.Parameter(down, requires_grad = False)
    try:
        return forward_native_grouped_mm(
            self,
            hidden_states,
            compact_top_k,
            top_k_weights,
        )
    finally:
        del self.gate_up_proj
        del self.down_proj
        self.num_experts = saved_n_experts


def _cached_dequant(module, attr_name, expert_idx, layer):
    """LRU dequant cache for one expert's weight. Hit returns the cached BF16
    tensor; miss dequants, stores, and evicts oldest. Cache cap is per-attribute
    per Gemma4TextExperts module so each layer maintains its own working set."""
    from bitsandbytes.functional import dequantize_4bit

    cache_attr = f"_unsloth_dequant_cache_{attr_name}"
    cache = getattr(module, cache_attr, None)
    if cache is None:
        from collections import OrderedDict

        cache = OrderedDict()
        setattr(module, cache_attr, cache)
    cached = cache.get(expert_idx, None)
    if cached is not None:
        cache.move_to_end(expert_idx)
        return cached
    w = dequantize_4bit(layer.weight.data, layer.weight.quant_state)
    cache[expert_idx] = w
    cap = int(os.environ.get("UNSLOTH_GEMMA4_MOE_4BIT_CACHE_SIZE", "8"))
    while len(cache) > cap:
        cache.popitem(last = False)
    return w


def _grouped_mm_forward_4bit_pt_compiled(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Active-only grouped forward using pure-PyTorch NF4 dequant + torch.compile.
    Pre-cached per-expert absmax means the per-forward path is bnb-free, so
    Inductor can fuse unpack + codebook lookup + stack into one Triton
    kernel."""
    from unsloth_zoo.temporary_patches.moe_utils import (
        forward_native_grouped_mm,
    )

    flat = top_k_index.reshape(-1)
    active_experts, inverse = torch.unique(flat, return_inverse = True)
    active_cpu = active_experts.tolist()
    n_active = len(active_cpu)

    codes = _NF4_CODES.to(hidden_states.device)
    dequant_fn = _get_compiled_pt_dequant_stack()

    gate_up = dequant_fn(self.gate_up_proj_4bit, active_cpu, codes)
    down = dequant_fn(self.down_proj_4bit, active_cpu, codes)

    compact_top_k = inverse.view_as(top_k_index)

    saved_n_experts = self.num_experts
    self.num_experts = n_active
    self.gate_up_proj = nn.Parameter(gate_up, requires_grad = False)
    self.down_proj = nn.Parameter(down, requires_grad = False)
    try:
        return forward_native_grouped_mm(
            self,
            hidden_states,
            compact_top_k,
            top_k_weights,
        )
    finally:
        del self.gate_up_proj
        del self.down_proj
        self.num_experts = saved_n_experts


def _grouped_mm_forward_4bit_static_bf16(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """First call: dequant every expert to a permanent (E, 2I, H) / (E, H, I)
    BF16 fused tensor stored on the module. Subsequent calls reuse them and
    skip dequant entirely. Speed-ceiling experiment: trades the 4-bit VRAM
    win for grouped_mm throughput."""
    from unsloth_zoo.temporary_patches.moe_utils import (
        forward_native_grouped_mm,
    )

    if not hasattr(self, "_unsloth_static_bf16_gate_up"):
        self._unsloth_static_bf16_gate_up = _dequant_stack(self.gate_up_proj_4bit)
        self._unsloth_static_bf16_down = _dequant_stack(self.down_proj_4bit)
    self.gate_up_proj = self._unsloth_static_bf16_gate_up
    self.down_proj = self._unsloth_static_bf16_down
    try:
        return forward_native_grouped_mm(
            self,
            hidden_states,
            top_k_index,
            top_k_weights,
        )
    finally:
        del self.gate_up_proj
        del self.down_proj


def _grouped_mm_forward_4bit_active_only_cached(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Active-only grouped forward with a per-module LRU cache of dequantized
    experts. Decode patterns where the same experts get hit across consecutive
    tokens skip the dequant on the second visit. Cache cap via
    UNSLOTH_GEMMA4_MOE_4BIT_CACHE_SIZE (default 8 per module)."""
    from unsloth_zoo.temporary_patches.moe_utils import (
        forward_native_grouped_mm,
    )

    flat = top_k_index.reshape(-1)
    active_experts, inverse = torch.unique(flat, return_inverse = True)
    active_cpu = active_experts.tolist()
    n_active = len(active_cpu)

    gate_up = torch.stack(
        [
            _cached_dequant(self, "gate_up", i, self.gate_up_proj_4bit[i])
            for i in active_cpu
        ],
        dim = 0,
    )
    down = torch.stack(
        [_cached_dequant(self, "down", i, self.down_proj_4bit[i]) for i in active_cpu],
        dim = 0,
    )

    compact_top_k = inverse.view_as(top_k_index)

    saved_n_experts = self.num_experts
    self.num_experts = n_active
    self.gate_up_proj = nn.Parameter(gate_up, requires_grad = False)
    self.down_proj = nn.Parameter(down, requires_grad = False)
    try:
        return forward_native_grouped_mm(
            self,
            hidden_states,
            compact_top_k,
            top_k_weights,
        )
    finally:
        del self.gate_up_proj
        del self.down_proj
        self.num_experts = saved_n_experts


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
        # Forward variant ladder (most specific wins). STATIC_BF16 is the
        # speed-ceiling variant; it overrides ACTIVE_ONLY/CACHE if set since
        # those become no-ops once weights are kept permanently dequantized.
        if (
            is_gemma4_moe_4bit_grouped_enabled()
            and is_gemma4_moe_4bit_grouped_static_bf16_enabled()
        ):
            _fwd = _grouped_mm_forward_4bit_static_bf16
        elif (
            is_gemma4_moe_4bit_grouped_enabled()
            and is_gemma4_moe_4bit_grouped_pt_dequant_enabled()
        ):
            for L in list(module.gate_up_proj_4bit) + list(module.down_proj_4bit):
                _ensure_pt_dequant_state(L)
            _fwd = _grouped_mm_forward_4bit_pt_compiled
        elif (
            is_gemma4_moe_4bit_grouped_enabled()
            and is_gemma4_moe_4bit_grouped_active_only_enabled()
            and is_gemma4_moe_4bit_grouped_cached_enabled()
        ):
            _fwd = _grouped_mm_forward_4bit_active_only_cached
        elif (
            is_gemma4_moe_4bit_grouped_enabled()
            and is_gemma4_moe_4bit_grouped_active_only_enabled()
        ):
            _fwd = _grouped_mm_forward_4bit_active_only
        elif is_gemma4_moe_4bit_grouped_enabled():
            _fwd = _grouped_mm_forward_4bit
        else:
            _fwd = _per_expert_forward
        module.forward = MethodType(_fwd, module)
        module._unsloth_gemma4_moe_4bit_swapped = True

        swapped += 1

    if swapped > 0 and torch.cuda.is_available():
        # Free the cached fused tensors so post-swap VRAM reflects 4-bit.
        torch.cuda.empty_cache()

    return swapped
