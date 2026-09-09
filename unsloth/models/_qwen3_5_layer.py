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
"""Two compiled regions per Qwen3.5 decoder layer for training.

Per layer the stock path runs ~6 separately compiled regions (input norm, mixer pieces,
post-attention norm, an MLP that breaks into three graphs around the LoRA linears) plus
eager residual adds, out_proj and reshapes. Every region call costs ~100 us of guard and
wrapper time on the CPU while the GPU sits idle, and there are 32 layers x forward,
recompute and backward per micro-step. Here a layer is:

    pre  = compiled( input_layernorm -> mixer input projections [-> q/k norms, RoPE] )
    mixer core, eager and unchanged: fla chunk_gated_delta_rule + gated RMSNorm, or SDPA
    post = compiled( out_proj [gate] -> residual add -> post_attention_layernorm
                     -> LoRA MLP -> residual add )

Weights are inputs, so one graph per region kind serves every layer. Every op keeps its
original dtype boundaries, so the numbers match the stock path. Training path only
(`past_key_values is None`); vanilla single-adapter LoRA or plain Linear; anything else
falls back to the original layer forward. `UNSLOTH_DISABLE_QWEN3_5_LAYER_FUSION=1`
turns it off.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._gated_delta_net import _fast_path_applicable as _gdn_applicable

__all__ = ["patch_qwen3_5_decoder_layers"]

_COMPILE_OPTIONS = {
    # Round every bf16 intermediate exactly where eager does (inductor otherwise keeps fused
    # intermediates in fp32), so the fused regions reproduce the stock path's numbers.
    "emulate_precision_casts": True,
    "epilogue_fusion": True,
    "max_autotune": False,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}


# ----------------------------------------------------------------------------- pieces
def _rmsnorm(x, weight, eps: float):
    # Qwen3_5RMSNorm: (x * rsqrt(mean(x^2) + eps)) * (1 + w), computed in fp32, cast back
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def _linear(x, W, A, B, scaling: float):
    """peft lora.Linear as unsloth's generated `Linear_peft_forward` computes it (vanilla LoRA,
    no dropout, no bias), or a plain F.linear when A is None."""
    if A is None:
        return F.linear(x, W)
    autocast = torch.is_autocast_enabled()
    if not autocast and x.dtype != W.dtype:
        x = x.to(W.dtype)
    result = F.linear(x, W)
    torch_result_dtype = result.dtype
    if not autocast:
        result, x = result.to(A.dtype), x.to(A.dtype)
    target_dtype = result.dtype
    xA = x.to(target_dtype) @ A.to(target_dtype).t()
    shape = result.shape
    out = torch.addmm(
        result.view(-1, shape[-1]),
        xA.view(-1, xA.shape[-1]),
        B.to(target_dtype).t(),
        alpha = scaling,
        beta = 1,
    ).view(shape)
    return out.to(torch_result_dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim = -1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim = -1)
    k_embed = torch.cat([k_embed, k_pass], dim = -1)
    return q_embed, k_embed


def _mlp_tail(mixer_out, residual, norm_w, eps: float, Wg, Ag, Bg, sg: float, Wu, Au, Bu, su: float,
              Wd, Ad, Bd, sd: float):
    hidden = residual + mixer_out
    x = _rmsnorm(hidden, norm_w, eps)
    mlp = _linear(F.silu(_linear(x, Wg, Ag, Bg, sg)) * _linear(x, Wu, Au, Bu, su), Wd, Ad, Bd, sd)
    return hidden + mlp


# ----------------------------------------------------------------------------- regions
@torch.compile(fullgraph = True, dynamic = True, options = _COMPILE_OPTIONS)
def _gdn_pre(hidden_states, norm_w, eps: float, attention_mask, w_qkv, w_z, w_b, w_a, conv_weight,
             A_log, dt_bias, key_dim: int, value_dim: int, head_k_dim: int, head_v_dim: int,
             conv_dim: int, conv_padding: int, n_rep: int):
    hidden_states = _rmsnorm(hidden_states, norm_w, eps)
    if attention_mask is not None:
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


@torch.compile(fullgraph = True, dynamic = True, options = _COMPILE_OPTIONS)
def _gdn_post(core_attn_out, residual, w_out, norm_w, eps: float, Wg, Ag, Bg, sg: float,
              Wu, Au, Bu, su: float, Wd, Ad, Bd, sd: float):
    batch_size, seq_len, _ = residual.shape
    out = F.linear(core_attn_out.reshape(batch_size, seq_len, -1), w_out)
    return _mlp_tail(out, residual, norm_w, eps, Wg, Ag, Bg, sg, Wu, Au, Bu, su, Wd, Ad, Bd, sd)


@torch.compile(fullgraph = True, dynamic = True, options = _COMPILE_OPTIONS)
def _attn_pre(hidden_states, norm_w, eps: float, Wq, Aq, Bq, sq: float, Wk, Ak, Bk, sk: float,
              Wv, Av, Bv, sv: float, qn_w, kn_w, cos, sin, head_dim: int):
    x = _rmsnorm(hidden_states, norm_w, eps)
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states, gate = torch.chunk(
        _linear(x, Wq, Aq, Bq, sq).view(*input_shape, -1, head_dim * 2), 2, dim = -1
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = _rmsnorm(query_states.view(hidden_shape), qn_w, eps).transpose(1, 2)
    key_states = _rmsnorm(_linear(x, Wk, Ak, Bk, sk).view(hidden_shape), kn_w, eps).transpose(1, 2)
    value_states = _linear(x, Wv, Av, Bv, sv).view(hidden_shape).transpose(1, 2)

    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if value_states.dtype != query_states.dtype:
        value_states = value_states.to(query_states.dtype)
    return query_states, key_states, value_states, gate


@torch.compile(fullgraph = True, dynamic = True, options = _COMPILE_OPTIONS)
def _attn_post(attn_output, gate, residual, Wo, Ao, Bo, so: float, norm_w, eps: float,
               Wg, Ag, Bg, sg: float, Wu, Au, Bu, su: float, Wd, Ad, Bd, sd: float):
    input_shape = residual.shape[:-1]
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    out = _linear(attn_output, Wo, Ao, Bo, so)
    return _mlp_tail(out, residual, norm_w, eps, Wg, Ag, Bg, sg, Wu, Au, Bu, su, Wd, Ad, Bd, sd)


# ----------------------------------------------------------------------------- module glue
def _lora_params(m):
    """(W, A, B, scaling) for a plain nn.Linear or a vanilla single-adapter peft lora.Linear in
    its normal state; None when the module is anything else right now."""
    if type(m) is nn.Linear:
        if m.bias is not None:
            return None
        return (m.weight, None, None, 1.0)
    base = getattr(m, "base_layer", None)
    if type(base) is not nn.Linear or base.bias is not None:
        return None
    if getattr(m, "disable_adapters", False) or getattr(m, "merged", False):
        return None
    active = m.active_adapters
    if len(active) != 1:
        return None
    name = active[0]
    if not hasattr(m, "lora_A") or name not in m.lora_A or name not in m.lora_B:
        return None
    lora_A, lora_B = m.lora_A[name], m.lora_B[name]
    if lora_B.bias is not None or name not in m.lora_dropout or name not in m.scaling:
        return None
    if name in getattr(m, "lora_variant", {}) or getattr(m, "use_dora", {}).get(name, False):
        return None
    if type(m.lora_dropout[name]) is not nn.Identity:
        return None
    return (base.weight, lora_A.weight, lora_B.weight, float(m.scaling[name]))


def _is_rmsnorm(m):
    return type(m).__name__ == "Qwen3_5RMSNorm" and isinstance(getattr(m, "weight", None), torch.Tensor) \
        and isinstance(getattr(m, "eps", None), float)


def _layer_static_ok(layer):
    if not (_is_rmsnorm(layer.input_layernorm) and _is_rmsnorm(layer.post_attention_layernorm)):
        return False
    mlp = getattr(layer, "mlp", None)
    if mlp is None or type(mlp).__name__ != "Qwen3_5MLP":
        return False
    act = getattr(mlp, "act_fn", None)
    # nn.SiLU, transformers' SiLUActivation (forward = F.silu) or F.silu itself
    if not (isinstance(act, nn.SiLU) or type(act).__name__ in ("SiLU", "SiLUActivation") or act is F.silu):
        return False
    if layer.layer_type == "linear_attention":
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None or type(gdn).__name__ != "Qwen3_5GatedDeltaNet" or not _gdn_applicable(gdn):
            return False
        if type(gdn.out_proj) is not nn.Linear or gdn.out_proj.bias is not None:
            return False
        return True
    if layer.layer_type == "full_attention":
        attn = getattr(layer, "self_attn", None)
        if attn is None or type(attn).__name__ != "Qwen3_5Attention":
            return False
        if not (_is_rmsnorm(attn.q_norm) and _is_rmsnorm(attn.k_norm)):
            return False
        impl = getattr(attn.config, "_attn_implementation", "eager")
        if impl == "eager":
            return False
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS[impl]
        except Exception:
            return False
        return True
    return False


def _fused_decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    **kwargs,
):
    if (
        past_key_values is not None
        or not getattr(self, "_unsloth_fused_ok", False)
        or kwargs.get("output_attentions", False)
    ):
        return self._unsloth_original_forward(
            hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            **kwargs,
        )
    mlp = self.mlp
    pg, pu, pd = _lora_params(mlp.gate_proj), _lora_params(mlp.up_proj), _lora_params(mlp.down_proj)
    norm2 = self.post_attention_layernorm
    norm1 = self.input_layernorm
    residual = hidden_states

    if self.layer_type == "linear_attention":
        gdn = self.linear_attn
        if pg is None or pu is None or pd is None:
            return self._unsloth_original_forward(
                hidden_states, position_embeddings = position_embeddings, attention_mask = attention_mask,
                position_ids = position_ids, past_key_values = past_key_values, **kwargs,
            )
        mask = attention_mask
        if mask is not None and not (mask.shape[1] > 1 and mask.shape[0] > 1):
            mask = None
        query, key, value, g, beta, z = _gdn_pre(
            hidden_states, norm1.weight, norm1.eps, mask,
            gdn.in_proj_qkv.weight, gdn.in_proj_z.weight, gdn.in_proj_b.weight, gdn.in_proj_a.weight,
            gdn.conv1d.weight, gdn.A_log, gdn.dt_bias,
            gdn.key_dim, gdn.value_dim, gdn.head_k_dim, gdn.head_v_dim, gdn.conv_dim,
            gdn.conv_kernel_size - 1, gdn.num_v_heads // gdn.num_k_heads,
        )
        core_attn_out, _ = gdn.chunk_gated_delta_rule(
            query, key, value, g = g, beta = beta, initial_state = None,
            output_final_state = False, use_qk_l2norm_in_kernel = True,
        )
        core_attn_out = gdn.norm(core_attn_out.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim))
        return _gdn_post(
            core_attn_out, residual, gdn.out_proj.weight, norm2.weight, norm2.eps,
            pg[0], pg[1], pg[2], pg[3], pu[0], pu[1], pu[2], pu[3], pd[0], pd[1], pd[2], pd[3],
        )

    attn = self.self_attn
    pq, pk, pv, po = _lora_params(attn.q_proj), _lora_params(attn.k_proj), _lora_params(attn.v_proj), _lora_params(attn.o_proj)
    if pg is None or pu is None or pd is None or pq is None or pk is None or pv is None or po is None:
        return self._unsloth_original_forward(
            hidden_states, position_embeddings = position_embeddings, attention_mask = attention_mask,
            position_ids = position_ids, past_key_values = past_key_values, **kwargs,
        )
    cos, sin = position_embeddings
    query_states, key_states, value_states, gate = _attn_pre(
        hidden_states, norm1.weight, norm1.eps,
        pq[0], pq[1], pq[2], pq[3], pk[0], pk[1], pk[2], pk[3], pv[0], pv[1], pv[2], pv[3],
        attn.q_norm.weight, attn.k_norm.weight, cos, sin, attn.head_dim,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]
    attn_output, _ = attention_interface(
        attn, query_states, key_states, value_states, attention_mask,
        dropout = 0.0 if not attn.training else attn.attention_dropout,
        scaling = attn.scaling,
        position_ids = position_ids,
        **kwargs,
    )
    return _attn_post(
        attn_output, gate, residual, po[0], po[1], po[2], po[3], norm2.weight, norm2.eps,
        pg[0], pg[1], pg[2], pg[3], pu[0], pu[1], pu[2], pu[3], pd[0], pd[1], pd[2], pd[3],
    )


def patch_qwen3_5_decoder_layers(model):
    """Install the fused two-region forward on every Qwen3_5DecoderLayer in `model`.
    Returns the number of layers that will take it."""
    if os.environ.get("UNSLOTH_DISABLE_QWEN3_5_LAYER_FUSION", "0") == "1":
        return 0
    n = 0
    for module in model.modules():
        cls = type(module)
        if cls.__name__ != "Qwen3_5DecoderLayer":
            continue
        if not hasattr(cls, "_unsloth_original_forward"):
            cls._unsloth_original_forward = cls.forward
            cls.forward = _fused_decoder_layer_forward
        try:
            module._unsloth_fused_ok = _layer_static_ok(module)
        except Exception:
            module._unsloth_fused_ok = False
        n += int(module._unsloth_fused_ok)
    layers = sum(1 for m in model.modules() if type(m).__name__ == "Qwen3_5DecoderLayer")
    if layers:
        print(f"Unsloth: Qwen3.5 fused two-region layer forward on {n}/{layers} decoder layers.")
    return n
