# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""fla chunk_gated_delta_rule as opaque op (training backward) so DeltaNet compiles as one graph."""

import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks


def _causal_depthwise_conv(x, weight):
    """F.conv1d(x, weight[:, None], groups = C, padding = K - 1)[..., :T] for x (B, C, T), weight
    (C, K), as K shifted multiply-adds accumulated in fp32. Inductor's convolution_backward
    lowering guards on the exact sequence length (a recompile per length); this stays dynamic."""
    seq_len, padding = x.shape[-1], weight.shape[-1] - 1
    xp = F.pad(x, (padding, 0)).float()
    w = weight.float()
    out = xp[:, :, 0:seq_len] * w[:, 0, None]
    for j in range(1, padding + 1):
        out = out + xp[:, :, j : j + seq_len] * w[:, j, None]
    return out.to(x.dtype)


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
            # The first call can come from a no_grad checkpoint forward, or a non-reentrant one
            # whose hooks pack the check's tensors; detached so a failed check leaves no cycle.
            with torch.inference_mode(False):
                with torch.no_grad():
                    inputs = [t.detach().clone() for t in make_inputs()]
                with torch.enable_grad(), saved_tensors_hooks(lambda t: t.detach(), lambda t: t):
                    _FLA = fns
                    _FLA = fns if _self_check(*inputs) else False
                    # Packed (padding-free) batches: B = 1 plus cu_seqlens, checked the same way.
                    _VARLEN[0] = bool(_FLA) and _self_check_varlen(*inputs)
        except Exception:
            _FLA = False
            _VARLEN[0] = False
    return _FLA


# True once the op matched fla's Function on a packed (cu_seqlens) batch too.
_VARLEN = [False]


def varlen_ok() -> bool:
    return _VARLEN[0]


def _varlen_kwargs(fn, cu_seqlens):
    """cu_seqlens (+ the chunk_indices fla's Function derives from it) for fla's fwd / bwd."""
    if cu_seqlens is None:
        return {}
    kwargs = {"cu_seqlens": cu_seqlens}
    if "chunk_indices" in inspect.signature(fn).parameters:
        from fla.ops.utils.index import prepare_chunk_indices
        kwargs["chunk_indices"] = prepare_chunk_indices(cu_seqlens, 64)
    return kwargs


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
        cu_seqlens: Optional[torch.Tensor] = None,
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
                **_varlen_kwargs(fwd, cu_seqlens),
            )
        # Fake metas are contiguous; keep the real outputs the same.
        return tuple(t.contiguous() for t in (o.to(q.dtype), qn, q_rstd, kn, k_rstd, g_cum, A))

    @_gdn_chunk.register_fake
    def _gdn_chunk_fake(
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens = None,
    ):
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
        cu_seqlens: Optional[torch.Tensor] = None,
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
                **_varlen_kwargs(bwd, cu_seqlens),
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
    def _gdn_chunk_bwd_fake(
        qn,
        q_rstd,
        kn,
        k_rstd,
        v,
        g_cum,
        beta,
        A,
        do,
        scale,
        cu_seqlens = None,
    ):
        return tuple(qn.new_empty(t.shape, dtype = t.dtype) for t in (qn, kn, v, g_cum, beta))

    def _gdn_chunk_setup(ctx, inputs, output):
        q, k, v, g, beta, scale = inputs[:6]
        cu_seqlens = inputs[6] if len(inputs) > 6 else None
        _, qn, q_rstd, kn, k_rstd, g_cum, A = output
        ctx.save_for_backward(qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, cu_seqlens)
        ctx.scale = scale
        ctx.dtypes = (q.dtype, k.dtype, g.dtype)

    def _gdn_chunk_backward(ctx, do, *_):
        *saved, cu_seqlens = ctx.saved_tensors
        saved = [t.contiguous() for t in saved]
        dq, dk, dv, dg, db = _gdn_chunk_bwd(*saved, do.contiguous(), ctx.scale, cu_seqlens)
        q_dtype, k_dtype, g_dtype = ctx.dtypes
        return dq.to(q_dtype), dk.to(k_dtype), dv, dg.to(g_dtype), db, None, None

    _gdn_chunk.register_autograd(_gdn_chunk_backward, setup_context = _gdn_chunk_setup)
except Exception:
    _gdn_chunk = None


def _self_check(q, k, v, g, beta):
    """Opaque op vs fla's own autograd.Function: outputs, fake metas and grads bit-equal."""
    if _gdn_chunk is None or not q.is_cuda:
        return False
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # Private generator: the default one feeds LoRA dropout, which checkpoint recompute replays.
    gen = torch.Generator(device = v.device).manual_seed(0)
    do = torch.randn(v.shape, dtype = v.dtype, device = v.device, generator = gen)
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


def _self_check_varlen(q, k, v, g, beta):
    """Packed batch: bit-equal to fla's Function; False if fla's fwd / bwd lack cu_seqlens."""
    if _gdn_chunk is None or not q.is_cuda:
        return False
    fwd, bwd, _, _ = _FLA
    for fn in (fwd, bwd):
        if "cu_seqlens" not in inspect.signature(fn).parameters:
            return False
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, T = q.shape[0], q.shape[1]
    flat = [t.reshape(1, B * T, *t.shape[2:]) for t in (q, k, v, g, beta)]
    total = B * T
    # Uneven documents, one shorter than a chunk and one crossing a chunk edge.
    cuts = sorted({0, min(17, total), min(total // 2 + 5, total), total})
    cu_seqlens = torch.tensor(cuts, device = q.device, dtype = torch.int32)
    gen = torch.Generator(device = v.device).manual_seed(1)
    do = torch.randn(flat[2].shape, dtype = v.dtype, device = v.device, generator = gen)
    ins = [t.contiguous().requires_grad_() for t in flat]
    ref, _ = chunk_gated_delta_rule(*ins, use_qk_l2norm_in_kernel = True, cu_seqlens = cu_seqlens)
    ref_g = torch.autograd.grad(ref, ins, do)
    ins2 = [t.detach().clone().requires_grad_() for t in ins]
    out = _gdn_chunk(*ins2, flat[0].shape[-1] ** -0.5, cu_seqlens)
    got_g = torch.autograd.grad(out[0], ins2, do)
    metas = [(o.shape, o.dtype) for o in out[1:]]
    want = [(torch.Size(s), d) for s, d in _chunk_meta(*flat[:4])]
    return (
        metas == want
        and torch.equal(out[0], ref)
        and all(torch.equal(a, b) for a, b in zip(got_g, ref_g))
    )
