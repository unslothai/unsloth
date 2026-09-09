# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Compiled input projection for Qwen3.5's GatedDeltaNet layers.

The generated `Qwen3_5GatedDeltaNet_forward` carries `@torch.compiler.disable` because it
calls fla's `chunk_gated_delta_rule`, an `autograd.Function` dynamo cannot trace. That
leaves the ~25 aten ops around the kernel (four projections, the depthwise conv + SiLU,
split/reshape, sigmoid, softplus gate, repeat_interleave) eager: launch-bound Python,
issued by 24 layers in forward, recompute and backward on every micro-step. Here those
ops form one `torch.compile`d region (weights are inputs, so one graph serves every
layer), the fla kernel is called between them exactly as before, and the gated RMSNorm +
output projection follow unchanged.

Training path only (`cache_params is None`). Anything else, a LoRA-wrapped projection, a
fused causal-conv1d being available, or an unexpected module layout, falls back to the
original forward. `UNSLOTH_DISABLE_GDN_FAST_FORWARD=1` turns it off.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["patch_gated_delta_net_fast_forward"]

_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "max_autotune": False,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}

_PATCHED_CLASS_NAMES = ("Qwen3_5GatedDeltaNet",)


@torch.compile(fullgraph = True, dynamic = True, options = _COMPILE_OPTIONS)
def _gdn_input_projection(
    hidden_states,
    attention_mask,
    w_qkv,
    w_z,
    w_b,
    w_a,
    conv_weight,
    A_log,
    dt_bias,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_dim: int,
    conv_padding: int,
    n_rep: int,
):
    if attention_mask is not None:
        # apply_mask_to_padding_states
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = F.linear(hidden_states, w_qkv).transpose(1, 2)
    z = F.linear(hidden_states, w_z).reshape(batch_size, seq_len, -1, head_v_dim)
    b = F.linear(hidden_states, w_b)
    a = F.linear(hidden_states, w_a)

    mixed_qkv = F.conv1d(mixed_qkv, conv_weight, None, 1, conv_padding, 1, conv_dim)[:, :, :seq_len]
    mixed_qkv = F.silu(mixed_qkv)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim = -1)
    query = query.reshape(batch_size, seq_len, -1, head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, head_v_dim)

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    if n_rep > 1:
        query = query.repeat_interleave(n_rep, dim = 2)
        key = key.repeat_interleave(n_rep, dim = 2)
    return query, key, value, g, beta, z


def _fast_path_applicable(self):
    if getattr(self, "causal_conv1d_fn", None) is not None:
        return False
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        m = getattr(self, name, None)
        if type(m) is not nn.Linear or m.bias is not None:
            return False
    conv = getattr(self, "conv1d", None)
    if type(conv) is not nn.Conv1d or conv.bias is not None or conv.groups != conv.in_channels:
        return False
    if conv.stride != (1,) or conv.dilation != (1,) or conv.padding != (self.conv_kernel_size - 1,):
        return False
    if not isinstance(getattr(self, "A_log", None), torch.Tensor) or not isinstance(getattr(self, "dt_bias", None), torch.Tensor):
        return False
    return True


def _gated_delta_net_fast_forward(self, hidden_states, cache_params = None, attention_mask = None):
    if cache_params is not None or not getattr(self, "_unsloth_gdn_fast_ok", False):
        return self._unsloth_original_forward(
            hidden_states, cache_params = cache_params, attention_mask = attention_mask
        )
    batch_size, seq_len, _ = hidden_states.shape
    mask = attention_mask
    if mask is not None and not (mask.shape[1] > 1 and mask.shape[0] > 1):
        mask = None
    query, key, value, g, beta, z = _gdn_input_projection(
        hidden_states,
        mask,
        self.in_proj_qkv.weight,
        self.in_proj_z.weight,
        self.in_proj_b.weight,
        self.in_proj_a.weight,
        self.conv1d.weight,
        self.A_log,
        self.dt_bias,
        self.key_dim,
        self.value_dim,
        self.head_k_dim,
        self.head_v_dim,
        self.conv_dim,
        self.conv_kernel_size - 1,
        self.num_v_heads // self.num_k_heads,
    )
    core_attn_out, _ = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g = g,
        beta = beta,
        initial_state = None,
        output_final_state = False,
        use_qk_l2norm_in_kernel = True,
    )
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def patch_gated_delta_net_fast_forward(model):
    """Install the compiled input projection on every Qwen3.5 GatedDeltaNet class in `model`.
    Returns the number of layers that will take the fast path."""
    if os.environ.get("UNSLOTH_DISABLE_GDN_FAST_FORWARD", "0") == "1":
        return 0
    n = 0
    for module in model.modules():
        cls = type(module)
        if cls.__name__ not in _PATCHED_CLASS_NAMES:
            continue
        if not hasattr(cls, "_unsloth_original_forward"):
            cls._unsloth_original_forward = cls.forward
            cls.forward = _gated_delta_net_fast_forward
        module._unsloth_gdn_fast_ok = _fast_path_applicable(module)
        n += int(module._unsloth_gdn_fast_ok)
    return n
