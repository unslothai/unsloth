# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Fused training forward for Qwen3.5 decoder layers.

The stock layer issues its norms, projections, conv, gates and residual adds as dozens of
eager ops and small compiled regions, so a LoRA step is CPU launch bound. Here a layer is

    pre  = compiled( input_layernorm -> mixer input projections [-> conv, gates | q/k norms] )
    mixer kernel, eager and unchanged: fla chunk_gated_delta_rule + gated RMSNorm, or attention
    [rope = compiled( RoPE ), attention layers]
    mid  = compiled( output projection -> residual add -> post_attention_layernorm )
    mlp  = compiled( MLP -> residual add )

With fla available, a GatedDeltaNet layer is a single compiled graph instead: the fla kernel
runs as an opaque custom op with its own backward (`unsloth::qwen3_5_gdn_chunk`), checked
bit-equal to fla's autograd.Function on the first real layer before it is used.

Weights are inputs, so one graph per region serves every layer. Training / no-cache path
only, with plain nn.Linear or vanilla single-adapter LoRA projections; anything else
(cache, packed batches, quantized or merged layers, DoRA, dropout) runs the stock forward.
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["patch_qwen3_5_decoder_layers"]

_BASE_OPTIONS = {
    "epilogue_fusion": True,
    "max_autotune": False,
    "shape_padding": True,
    "triton.cudagraphs": False,
}
# Regions replacing eager ops round every bf16 intermediate like eager does; the MLP and RoPE
# regions replace compiled graphs, so they keep inductor's defaults.
_EAGER_LIKE_OPTIONS = dict(_BASE_OPTIONS, emulate_precision_casts = True)
_COMPILED_LIKE_OPTIONS = dict(_BASE_OPTIONS)
# dynamic=True fails in inductor value-range analysis on torch 2.13; automatic dynamic marks
# only the sequence dim after the first recompile.
_DYNAMIC = None

# Packed batches: the stock forward hands these to the conv and the fla kernel.
_PACKED_KWARGS = ("cu_seq_lens_q", "cu_seqlens", "seq_idx")


def _rmsnorm(x, weight, eps: float):
    # Qwen3_5RMSNorm: fp32 math, (1 + w) scale, cast back
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def _linear(x, p):
    """nn.Linear, or peft lora.Linear as unsloth's generated `Linear_peft_forward` computes it."""
    W, A, B, scaling = p
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


def _gdn_pre_impl(
    hidden_states,
    norm_w,
    eps: float,
    mask,
    p_qkv,
    p_z,
    p_b,
    p_a,
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
    if mask is not None:
        # The stock path masks after the norm; RMSNorm(0) == 0 and the mask is 0/1, so masking
        # first is the same result.
        hidden_states = (hidden_states * mask[:, :, None]).to(hidden_states.dtype)
    x = _rmsnorm(hidden_states, norm_w, eps)
    batch_size, seq_len, _ = x.shape

    mixed_qkv = _linear(x, p_qkv).transpose(1, 2)
    z = _linear(x, p_z).reshape(batch_size, seq_len, -1, head_v_dim)
    b = _linear(x, p_b)
    a = _linear(x, p_a)

    mixed_qkv = F.conv1d(mixed_qkv, conv_weight, None, 1, conv_padding, 1, conv_dim)[:, :, :seq_len]
    mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)
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


_gdn_pre = torch.compile(
    _gdn_pre_impl, fullgraph = True, dynamic = _DYNAMIC, options = _EAGER_LIKE_OPTIONS
)


# fla's chunk_gated_delta_rule is an autograd.Function dynamo cannot trace. As an opaque op with
# a registered backward (training case: no initial / final state, no cu_seqlens, q/k l2norm in
# kernel) the whole GatedDeltaNet layer compiles as one graph. fwd/bwd call the same fla
# functions ChunkGatedDeltaRuleFunction does, so the kernel numbers are unchanged.
_FLA = None


def _fla_ops(make_inputs):
    """fla's (fwd, bwd, l2norm_fwd, l2norm_bwd) once the opaque op matched fla's own Function on
    the first real layer's inputs, else False. Checked on real shapes and run through fla's
    Function first, so its pinned autotune configs are the ones the stock path would pick."""
    global _FLA
    if _FLA is None:
        _FLA = False
        try:
            import fla.ops.gated_delta_rule.chunk as C
            fns = (
                C.chunk_gated_delta_rule_fwd,
                C.chunk_gated_delta_rule_bwd,
                C.l2norm_fwd,
                C.l2norm_bwd,
            )
            # The first call can come from a no_grad checkpoint forward.
            with torch.inference_mode(False):
                with torch.no_grad():
                    inputs = [t.detach().clone() for t in make_inputs()]
                with torch.enable_grad():
                    _FLA = fns
                    _FLA = fns if _self_check(*inputs) else False
        except Exception:
            _FLA = False
    return _FLA


def _chunk_meta(q, k, v, g):
    # (qn, q_rstd, kn, k_rstd, g_cumsum, A), as fla 0.5 allocates them
    return (
        (q.shape, q.dtype),
        (q.shape[:-1], torch.float32),
        (k.shape, k.dtype),
        (k.shape[:-1], torch.float32),
        (g.shape, torch.float32),
        ((*k.shape[:-1], 64), k.dtype),
    )


try:

    @torch.library.custom_op("unsloth::qwen3_5_gdn_chunk", mutates_args = ())
    def _gdn_chunk(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fwd, _, l2_fwd, _ = _FLA
        q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
        with torch.cuda.device(q.device):
            qn, q_rstd = l2_fwd(q)
            kn, k_rstd = l2_fwd(k)
            g_cum, o, A, _, _, _ = fwd(
                q = qn,
                k = kn,
                v = v,
                g = g,
                beta = beta,
                scale = scale,
                initial_state = None,
                output_final_state = False,
            )
        # Fake metas are contiguous; keep the real outputs the same.
        return tuple(t.contiguous() for t in (o.to(q.dtype), qn, q_rstd, kn, k_rstd, g_cum, A))

    @_gdn_chunk.register_fake
    def _(q, k, v, g, beta, scale):
        o = v.new_empty(v.shape, dtype = q.dtype)
        return (o,) + tuple(
            q.new_empty(shape, dtype = dtype) for shape, dtype in _chunk_meta(q, k, v, g)
        )

    @torch.library.custom_op("unsloth::qwen3_5_gdn_chunk_bwd", mutates_args = ())
    def _gdn_chunk_bwd(
        qn: torch.Tensor,
        q_rstd: torch.Tensor,
        kn: torch.Tensor,
        k_rstd: torch.Tensor,
        v: torch.Tensor,
        g_cum: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        do: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, bwd, _, l2_bwd = _FLA
        v, beta, do = v.contiguous(), beta.contiguous(), do.contiguous()
        with torch.cuda.device(qn.device):
            dq, dk, dv, db, dg, _, _, _ = bwd(
                q = qn,
                k = kn,
                v = v,
                g = g_cum,
                beta = beta,
                A = A,
                scale = scale,
                initial_state = None,
                do = do,
                dht = None,
            )
            dq = l2_bwd(qn, q_rstd, dq)
            dk = l2_bwd(kn, k_rstd, dk)
        out = (
            dq.to(qn.dtype),
            dk.to(kn.dtype),
            dv.to(v.dtype),
            dg.to(g_cum.dtype),
            db.to(beta.dtype),
        )
        return tuple(t.contiguous() for t in out)

    @_gdn_chunk_bwd.register_fake
    def _(qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, do, scale):
        return tuple(qn.new_empty(t.shape, dtype = t.dtype) for t in (qn, kn, v, g_cum, beta))

    def _gdn_chunk_setup(ctx, inputs, output):
        q, k, v, g, beta, scale = inputs
        _, qn, q_rstd, kn, k_rstd, g_cum, A = output
        ctx.save_for_backward(qn, q_rstd, kn, k_rstd, v, g_cum, beta, A)
        ctx.scale = scale
        ctx.dtypes = (q.dtype, k.dtype, g.dtype)

    def _gdn_chunk_backward(ctx, do, *_):
        # Traced into the backward graph: contiguous here so inductor hands the op the layout
        # its fake was traced with.
        saved = [t.contiguous() for t in ctx.saved_tensors]
        dq, dk, dv, dg, db = _gdn_chunk_bwd(*saved, do.contiguous(), ctx.scale)
        q_dtype, k_dtype, g_dtype = ctx.dtypes
        return dq.to(q_dtype), dk.to(k_dtype), dv, dg.to(g_dtype), db, None

    _gdn_chunk.register_autograd(_gdn_chunk_backward, setup_context = _gdn_chunk_setup)
except Exception:
    _gdn_chunk = None


def _self_check(q, k, v, g, beta):
    """Opaque op vs fla's own autograd.Function: outputs, fake metas and grads bit-equal."""
    if _gdn_chunk is None or not q.is_cuda:
        return False
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    do = torch.randn_like(v)
    ins = [t.contiguous().requires_grad_() for t in (q, k, v, g, beta)]
    ref, _ = chunk_gated_delta_rule(*ins, use_qk_l2norm_in_kernel = True)
    ref_g = torch.autograd.grad(ref, ins, do)
    ins2 = [t.detach().clone().requires_grad_() for t in ins]
    out = _gdn_chunk(*ins2, q.shape[-1] ** -0.5)
    got_g = torch.autograd.grad(out[0], ins2, do)
    metas = [(o.shape, o.dtype) for o in out[1:]]
    want = [(torch.Size(s), d) for s, d in _chunk_meta(q, k, v, g)]
    return (
        metas == want
        and torch.equal(out[0], ref)
        and all(torch.equal(a, b) for a, b in zip(got_g, ref_g))
    )


def _gated_rmsnorm(x, gate, weight, eps: float, round_after_weight: bool):
    # Qwen3_5RMSNormGated rounds to the input dtype before the weight; fla's FusedRMSNormGated
    # stays in fp32 until the output.
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim = True) + eps)
    y = weight * y.to(dtype) if round_after_weight else y * weight.float()
    return (y * F.silu(gate.float())).to(dtype)


@torch.compile(fullgraph = True, dynamic = _DYNAMIC, options = _EAGER_LIKE_OPTIONS)
def _gdn_layer(
    hidden_states,
    norm_w,
    eps: float,
    mask,
    p_qkv,
    p_z,
    p_b,
    p_a,
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
    gn_w,
    gn_eps: float,
    gn_round: bool,
    p_out,
    norm2_w,
    eps2: float,
    p_g,
    p_u,
    p_d,
):
    query, key, value, g, beta, z = _gdn_pre_impl(
        hidden_states,
        norm_w,
        eps,
        mask,
        p_qkv,
        p_z,
        p_b,
        p_a,
        conv_weight,
        A_log,
        dt_bias,
        key_dim,
        value_dim,
        head_k_dim,
        head_v_dim,
        conv_dim,
        conv_padding,
        n_rep,
    )
    query, key, value, g, beta = (t.contiguous() for t in (query, key, value, g, beta))
    core = torch.ops.unsloth.qwen3_5_gdn_chunk(query, key, value, g, beta, head_k_dim**-0.5)[0]
    core = _gated_rmsnorm(
        core.reshape(-1, head_v_dim), z.reshape(-1, head_v_dim), gn_w, gn_eps, gn_round
    )
    batch_size, seq_len, _ = hidden_states.shape
    hidden = hidden_states + _linear(core.reshape(batch_size, seq_len, -1), p_out)
    x = _rmsnorm(hidden, norm2_w, eps2)
    return hidden + _linear(F.silu(_linear(x, p_g)) * _linear(x, p_u), p_d)


@torch.compile(fullgraph = True, dynamic = _DYNAMIC, options = _EAGER_LIKE_OPTIONS)
def _gdn_mid(core_attn_out, residual, p_out, norm_w, eps: float):
    batch_size, seq_len, _ = residual.shape
    hidden = residual + _linear(core_attn_out.reshape(batch_size, seq_len, -1), p_out)
    return hidden, _rmsnorm(hidden, norm_w, eps)


@torch.compile(fullgraph = True, dynamic = _DYNAMIC, options = _EAGER_LIKE_OPTIONS)
def _attn_pre(hidden_states, norm_w, eps: float, p_q, p_k, p_v, qn_w, kn_w, head_dim: int):
    x = _rmsnorm(hidden_states, norm_w, eps)
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    query_states, gate = torch.chunk(
        _linear(x, p_q).view(*input_shape, -1, head_dim * 2), 2, dim = -1
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = _rmsnorm(query_states.view(hidden_shape), qn_w, eps).transpose(1, 2)
    key_states = _rmsnorm(_linear(x, p_k).view(hidden_shape), kn_w, eps).transpose(1, 2)
    value_states = _linear(x, p_v).view(hidden_shape).transpose(1, 2)
    return query_states, key_states, value_states, gate


@torch.compile(fullgraph = True, dynamic = _DYNAMIC, options = _COMPILED_LIKE_OPTIONS)
def _attn_rope(q, k, v, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim = -1)
    k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim = -1)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)
    return q, k, v


@torch.compile(fullgraph = True, dynamic = _DYNAMIC, options = _EAGER_LIKE_OPTIONS)
def _attn_mid(attn_output, gate, residual, p_o, norm_w, eps: float):
    input_shape = residual.shape[:-1]
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    hidden = residual + _linear(attn_output * torch.sigmoid(gate), p_o)
    return hidden, _rmsnorm(hidden, norm_w, eps)


@torch.compile(fullgraph = True, dynamic = _DYNAMIC, options = _COMPILED_LIKE_OPTIONS)
def _mlp_block(x, hidden, p_g, p_u, p_d):
    return hidden + _linear(F.silu(_linear(x, p_g)) * _linear(x, p_u), p_d)


def _lora_params(m):
    """(W, A, B, scaling) for a plain nn.Linear or a vanilla single-adapter peft lora.Linear;
    None for anything else. Cached per module, re-validated on the flags that flip at run time."""
    cached = m.__dict__.get("_unsloth_lora_params_cache")
    if cached is not None:
        params, adapters = cached
        if not m.disable_adapters and not m.merged and m.active_adapters == adapters:
            return params
    params = _lora_params_uncached(m)
    if params is not None and type(m) is not nn.Linear:
        m._unsloth_lora_params_cache = (params, list(m.active_adapters))
    return params


def _lora_params_uncached(m):
    if type(m) is nn.Linear:
        return None if m.bias is not None else (m.weight, None, None, 1.0)
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


def _block_type(layer):
    # transformers < 5.15 `layer_type`, >= 5.15 `block_type`
    return getattr(layer, "layer_type", None) or getattr(layer, "block_type", None)


def _chunk_kernel(gdn):
    # transformers < 5.15 binds the kernel on the module; >= 5.15 calls a module-level function.
    fn = getattr(gdn, "chunk_gated_delta_rule", None)
    if callable(fn):
        return fn
    fn = getattr(sys.modules.get(type(gdn).__module__), "torch_chunk_gated_delta_rule", None)
    return fn if callable(fn) else None


def _masks_any_batch(gdn):
    # transformers < 5.15 masks only when batch > 1 and seq > 1; >= 5.15 whenever a mask is given.
    fn = getattr(sys.modules.get(type(gdn).__module__), "apply_mask_to_padding_states", None)
    if fn is None:
        return None
    out = fn(torch.ones(1, 2, 1), torch.zeros(1, 2, dtype = torch.long))
    return bool((out == 0).all())


def _gated_norm_spec(m):
    """(round_after_weight, eps) for the GDN output norm, or None if it is not one we replicate."""
    name = type(m).__name__
    if not isinstance(getattr(m, "weight", None), torch.Tensor):
        return None
    if name == "Qwen3_5RMSNormGated" and getattr(m, "activation", "silu") in ("silu", "swish"):
        eps = getattr(m, "variance_epsilon", None)
        return (True, eps) if isinstance(eps, float) else None
    if name == "FusedRMSNormGated" and getattr(m, "activation", None) in ("silu", "swish"):
        eps = getattr(m, "eps", None)
        if getattr(m, "bias", None) is None and isinstance(eps, float):
            return (False, eps)
    return None


def _is_rmsnorm(m):
    return (
        type(m).__name__ == "Qwen3_5RMSNorm"
        and isinstance(getattr(m, "weight", None), torch.Tensor)
        and isinstance(getattr(m, "eps", None), float)
    )


def _layer_static_ok(layer):
    if not (_is_rmsnorm(layer.input_layernorm) and _is_rmsnorm(layer.post_attention_layernorm)):
        return False
    mlp = getattr(layer, "mlp", None)
    act = getattr(mlp, "act_fn", None)
    if type(mlp).__name__ != "Qwen3_5MLP" or not (
        isinstance(act, nn.SiLU)
        or type(act).__name__ in ("SiLU", "SiLUActivation")
        or act is F.silu
    ):
        return False
    kind = _block_type(layer)
    if kind == "linear_attention":
        gdn = getattr(layer, "linear_attn", None)
        if type(gdn).__name__ != "Qwen3_5GatedDeltaNet" or _chunk_kernel(gdn) is None:
            return False
        if getattr(gdn, "activation", "silu") not in ("silu", "swish"):
            return False
        conv = getattr(gdn, "conv1d", None)
        if type(conv) is not nn.Conv1d or conv.bias is not None or conv.groups != conv.in_channels:
            return False
        if (
            conv.stride != (1,)
            or conv.dilation != (1,)
            or conv.padding != (gdn.conv_kernel_size - 1,)
        ):
            return False
        masks_any = _masks_any_batch(gdn)
        if masks_any is None:
            return False
        layer._unsloth_masks_any_batch = masks_any
        layer._unsloth_gated_norm = _gated_norm_spec(getattr(gdn, "norm", None))
        return isinstance(getattr(gdn, "A_log", None), torch.Tensor) and isinstance(
            getattr(gdn, "dt_bias", None), torch.Tensor
        )
    if kind == "full_attention":
        attn = getattr(layer, "self_attn", None)
        if type(attn).__name__ != "Qwen3_5Attention":
            return False
        if not (_is_rmsnorm(attn.q_norm) and _is_rmsnorm(attn.k_norm)):
            return False
        impl = getattr(attn.config, "_attn_implementation", "eager")
        if impl == "eager":
            return False
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        return impl in ALL_ATTENTION_FUNCTIONS
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
    params = None
    if (
        past_key_values is None
        and getattr(self, "_unsloth_fused_ok", False)
        and not kwargs.get("output_attentions", False)
        and not any(kwargs.get(k) is not None for k in _PACKED_KWARGS)
    ):
        is_gdn = _block_type(self) == "linear_attention"
        mixer = self.linear_attn if is_gdn else self.self_attn
        names = (
            ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
            if is_gdn
            else ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        params = [_lora_params(getattr(mixer, n)) for n in names]
        params += [
            _lora_params(getattr(self.mlp, n)) for n in ("gate_proj", "up_proj", "down_proj")
        ]
        if any(p is None for p in params):
            params = None
    if params is None:
        return self._unsloth_original_forward(
            hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            **kwargs,
        )
    *mixer_params, p_g, p_u, p_d = params
    norm1, norm2 = self.input_layernorm, self.post_attention_layernorm
    residual = hidden_states

    if is_gdn:
        gdn = mixer
        p_qkv, p_z, p_b, p_a, p_out = mixer_params
        mask = attention_mask
        if mask is not None and mask.dim() != 2:
            mask = None
        if mask is not None and not self._unsloth_masks_any_batch:
            if not (mask.shape[1] > 1 and mask.shape[0] > 1):
                mask = None
        gn = self._unsloth_gated_norm
        pre_args = (
            hidden_states,
            norm1.weight,
            norm1.eps,
            mask,
            p_qkv,
            p_z,
            p_b,
            p_a,
            gdn.conv1d.weight,
            gdn.A_log,
            gdn.dt_bias,
            gdn.key_dim,
            gdn.value_dim,
            gdn.head_k_dim,
            gdn.head_v_dim,
            gdn.conv_dim,
            gdn.conv_kernel_size - 1,
            gdn.num_v_heads // gdn.num_k_heads,
        )
        if gn is not None and _fla_ops(lambda: _gdn_pre(*pre_args)[:5]):
            return _gdn_layer(
                *pre_args,
                gdn.norm.weight,
                gn[1],
                gn[0],
                p_out,
                norm2.weight,
                norm2.eps,
                p_g,
                p_u,
                p_d,
            )
        query, key, value, g, beta, z = _gdn_pre(*pre_args)
        core_attn_out, _ = _chunk_kernel(gdn)(
            query,
            key,
            value,
            g = g,
            beta = beta,
            initial_state = None,
            output_final_state = False,
            use_qk_l2norm_in_kernel = True,
        )
        core_attn_out = gdn.norm(
            core_attn_out.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim)
        )
        hidden, x = _gdn_mid(core_attn_out, residual, p_out, norm2.weight, norm2.eps)
        return _mlp_block(x, hidden, p_g, p_u, p_d)

    attn = mixer
    p_q, p_k, p_v, p_o = mixer_params
    query_states, key_states, value_states, gate = _attn_pre(
        hidden_states,
        norm1.weight,
        norm1.eps,
        p_q,
        p_k,
        p_v,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.head_dim,
    )
    cos, sin = position_embeddings
    query_states, key_states, value_states = _attn_rope(
        query_states, key_states, value_states, cos, sin
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attn_output, _ = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation](
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout = 0.0 if not attn.training else attn.attention_dropout,
        scaling = attn.scaling,
        position_ids = position_ids,
        **kwargs,
    )
    hidden, x = _attn_mid(attn_output, gate, residual, p_o, norm2.weight, norm2.eps)
    return _mlp_block(x, hidden, p_g, p_u, p_d)


def patch_qwen3_5_decoder_layers(model):
    """Install the fused forward on every Qwen3_5DecoderLayer; returns how many layers take it."""
    n = layers = 0
    for module in model.modules():
        cls = type(module)
        if cls.__name__ != "Qwen3_5DecoderLayer":
            continue
        layers += 1
        if "_unsloth_original_forward" not in cls.__dict__:
            cls._unsloth_original_forward = cls.forward
            cls.forward = _fused_decoder_layer_forward
        try:
            module._unsloth_fused_ok = _layer_static_ok(module)
        except Exception:
            module._unsloth_fused_ok = False
        n += module._unsloth_fused_ok
    if layers:
        print(f"Unsloth: Qwen3.5 fused layer forward on {n}/{layers} decoder layers.")
    return n
