# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MOSS-Image blocks inside the regional compile: one q/k/v GEMM, and int8 Linears through ``int8_linear``.

ComfyUI's repacked MOSS files fuse ``proj.q/k/v`` into one ``proj.qkv`` Linear ("one GEMM and one activation
quantization instead of three"). Here the three bias-free int8 projections become one Linear over the concatenated
output rows, built at load from the checkpoint's own tensors: codes and per-row scales concatenate; ConvRot parts
rotate the SAME input at one group, and the rotation acts on the shared input axis, so the fused Linear is itself a
ConvRotLinear (one rotation + one act quant + one GEMM for q, k and v). Per-row scales make the fused output equal the
three separate ones. The parts' weights become views of the fused one, so no memory is added. The block reads the
fused Linear only inside the compiled forward and only while ``proj.q/k/v`` are still the modules it was built from
(a LoRA wrapper keeps stock). bf16 keeps three GEMMs: one fused GEMM measured neutral there.

int8 (plain or ConvRot) Linears also run through ``diffusion_int8_fused.int8_linear``: the module's own rotation, the
probed Triton act quant (bit-identical to torchao's) and ``_int_mm`` with torchao's epilogue, i.e. torchao's eager math
where the stock compiled path is Inductor's fp32 chain.

Measured on an exclusive B200 (1024x1024, 30 steps, CFG 3, CUDA graphs, s/step): INT8-ConvRot 0.0769-0.0778 ->
0.0752 with the runner -> 0.0722-0.0726 with the fused QKV too; plain INT8 0.0700 -> 0.0673-0.0678. Peak unchanged.

``UNSLOTH_DIFFUSION_MOSS_QK_ROPE=1`` (opt-in) also runs the q/k RMSNorm + split-half RoPE + [B, H, S, D] relayout as
one kernel below; it bought 0.7% on bf16 and nothing on int8, so it stays off.

Kill switch: ``UNSLOTH_DIFFUSION_MOSS_FUSED=0`` (``UNSLOTH_DIFFUSION_MOSS_FUSED_QKV=0`` keeps only the int8 runner).
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any

MOSS_FUSED_ENV = "UNSLOTH_DIFFUSION_MOSS_FUSED"
MOSS_FUSED_QKV_ENV = "UNSLOTH_DIFFUSION_MOSS_FUSED_QKV"
_LOCK = threading.Lock()


def _off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("0", "off", "false", "no")


def _on(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "on", "true", "yes")


def moss_fused_disabled() -> bool:
    return _off(MOSS_FUSED_ENV)


def _blocks(root: Any) -> list:
    from .diffusion_moss_image import _Block
    return [m for m in root.modules() if type(m) is _Block]


def _int8_runnable(linear: Any) -> bool:
    from torch import nn

    from .diffusion_convrot import is_rotated_linear
    from .diffusion_int8_fused import _I8_GEMM_MARK, _plain_int8_weight

    d = getattr(linear, "__dict__", {})
    return (
        (type(linear) is nn.Linear or is_rotated_linear(linear))
        and "forward" not in d
        and _I8_GEMM_MARK not in d
        and _plain_int8_weight(linear.weight)
    )


def install(
    transformer: Any,
    logger: Any = None,
    offload_active: bool = False,
) -> dict:
    """Before the first compile; deferred to the first forward when the weights are not on the GPU yet. A streamed
    (offloaded) denoiser keeps the stock projections: its hooks own each Linear's weights."""
    result = {"fused_qkv": 0, "int8_runner": 0, "qk_rope": 0}
    if (
        moss_fused_disabled()
        or offload_active
        or type(transformer).__name__ != "MOSSImageTransformer"
    ):
        uninstall(transformer)
        return result
    from .diffusion_int8_fused import resident_cuda_device, run_on_first_call

    if resident_cuda_device(transformer) is not None:
        return install_modules(transformer, logger)
    run_on_first_call(transformer, "moss_fused", lambda t: install_modules(t, logger))
    return {"fused_qkv": None, "int8_runner": None}


def install_modules(root: Any, logger: Any = None) -> dict:
    from .diffusion_int8_fused import arm_linear, int8_linear, resident_cuda_device
    import torch

    from .diffusion_moss_image import FUSED_QKV_ATTR, LINEAR_RUNNER_ATTR, QK_ROPE_ATTR
    from .diffusion_zimage_fused import _fuse_linears, _share_storage

    result = {"fused_qkv": 0, "int8_runner": 0, "qk_rope": 0}
    dev = resident_cuda_device(root)
    if dev is None or moss_fused_disabled():
        return result
    blocks = _blocks(root)
    int8 = any(_int8_runnable(b.proj["q"]) for b in blocks)
    armed = arm_linear(dev) if int8 else False
    fuse = not _off(MOSS_FUSED_QKV_ENV)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    rope_op = qk_rope_op() if _on(MOSS_QK_ROPE_ENV) and qk_rope_device_ok(index) else None
    with _LOCK:
        for block in blocks:
            norms_ok = all(
                n.weight.dtype == torch.bfloat16 and n.weight.numel() == 128
                for n in (block.q_norm, block.k_norm)
            )
            if rope_op is not None and norms_ok:
                block.__dict__[QK_ROPE_ATTR] = rope_op
                result["qk_rope"] += 1
            else:
                block.__dict__.pop(QK_ROPE_ATTR, None)
            runner = int8_linear if int8 and _int8_runnable(block.proj["q"]) else None
            if runner is not None:
                block.__dict__[LINEAR_RUNNER_ATTR] = runner
                result["int8_runner"] += 1
            if not fuse or runner is None:
                # bf16: one q/k/v GEMM measured neutral (B200, 0.0879 vs 0.0881 s/step, inside the A/A spread)
                block.__dict__.pop(FUSED_QKV_ATTR, None)
                continue
            parts = [block.proj[n] for n in ("q", "k", "v")]
            rec = block.__dict__.get(FUSED_QKV_ATTR)
            if rec is None or any(a is not b for a, b in zip(rec[1:4], parts)):
                fused = _fuse_linears(parts)
                if fused is None or not _share_storage(fused, parts):
                    continue
                block.__dict__[FUSED_QKV_ATTR] = (fused, *parts, runner)
            result["fused_qkv"] += 1
    if logger is not None:
        logger.info(
            "diffusion.moss_fused: one q/k/v GEMM on %d of %d block(s); int8 runner on %d (act quant kernel %s); "
            "fused q/k norm + RoPE on %d",
            result["fused_qkv"],
            len(blocks),
            result["int8_runner"],
            "probed" if armed else "torchao's",
            result["qk_rope"],
        )
    return result


def uninstall(transformer: Any = None) -> None:
    """Drop the fused Linears (the per-row views stay valid) and any pending install."""
    if transformer is None:
        return
    from .diffusion_int8_fused import cancel_first_call
    from .diffusion_moss_image import FUSED_QKV_ATTR, LINEAR_RUNNER_ATTR, QK_ROPE_ATTR

    cancel_first_call(transformer, "moss_fused")
    try:
        blocks = _blocks(transformer)
    except Exception:  # noqa: BLE001
        return
    with _LOCK:
        for block in blocks:
            block.__dict__.pop(FUSED_QKV_ATTR, None)
            block.__dict__.pop(LINEAR_RUNNER_ATTR, None)
            block.__dict__.pop(QK_ROPE_ATTR, None)


# Fuses q/k RMSNorm + RoPE + layout copy into one pass; keeps every eager bf16 rounding of the port.
MOSS_QK_ROPE_ENV = "UNSLOTH_DIFFUSION_MOSS_QK_ROPE"
_QK_ROPE_OP = "moss_qk_norm_rope"
_HEADS_PER_PROGRAM = 8


@functools.lru_cache(maxsize = 1)
def _qk_rope_kernel() -> Any:
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001
        return None

    @triton.jit
    def _norm_rope(
        x_ptr,
        w_ptr,
        o_ptr,
        base,
        out,
        h,
        j,
        c1,
        s1,
        c2,
        s2,
        mask,
        EPS: tl.constexpr,
        D: tl.constexpr,
    ):
        # rotate(x) = [-x2, x1]; each product and sum rounded to bf16 like eager
        x1 = tl.load(x_ptr + base + h * D + j, mask = mask, other = 0.0).to(tl.float32)
        x2 = tl.load(x_ptr + base + h * D + D // 2 + j, mask = mask, other = 0.0).to(tl.float32)
        r = 1.0 / tl.sqrt_rn((tl.sum(x1 * x1, axis = 1) + tl.sum(x2 * x2, axis = 1)) / D + EPS)
        w1 = tl.load(w_ptr + j).to(tl.float32)
        w2 = tl.load(w_ptr + D // 2 + j).to(tl.float32)
        n1 = ((x1 * r[:, None]).to(tl.bfloat16).to(tl.float32) * w1).to(tl.bfloat16).to(tl.float32)
        n2 = ((x2 * r[:, None]).to(tl.bfloat16).to(tl.float32) * w2).to(tl.bfloat16).to(tl.float32)
        o1 = (n1 * c1).to(tl.bfloat16).to(tl.float32) + (-n2 * s1).to(tl.bfloat16).to(tl.float32)
        o2 = (n2 * c2).to(tl.bfloat16).to(tl.float32) + (n1 * s2).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + out + j, o1.to(tl.bfloat16), mask = mask)
        tl.store(o_ptr + out + D // 2 + j, o2.to(tl.bfloat16), mask = mask)

    @triton.jit
    def kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        qw_ptr,
        kw_ptr,
        cos_ptr,
        sin_ptr,
        oq_ptr,
        ok_ptr,
        ov_ptr,
        S,
        H,
        sqb,
        sqs,
        skb,
        sks,
        svb,
        svs,
        scs,
        EPS: tl.constexpr,
        D: tl.constexpr,
        HB: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // S
        s = row % S
        h = tl.program_id(1) * HB + tl.arange(0, HB)[:, None]
        j = tl.arange(0, D // 2)[None, :]
        mask = h < H
        c1 = tl.load(cos_ptr + s * scs + j).to(tl.float32)
        s1 = tl.load(sin_ptr + s * scs + j).to(tl.float32)
        c2 = tl.load(cos_ptr + s * scs + D // 2 + j).to(tl.float32)
        s2 = tl.load(sin_ptr + s * scs + D // 2 + j).to(tl.float32)
        out = ((b * H + h) * S + s) * D
        _norm_rope(
            q_ptr, qw_ptr, oq_ptr, b * sqb + s * sqs, out, h, j, c1, s1, c2, s2, mask, EPS, D
        )
        _norm_rope(
            k_ptr, kw_ptr, ok_ptr, b * skb + s * sks, out, h, j, c1, s1, c2, s2, mask, EPS, D
        )
        jj = tl.arange(0, D)[None, :]
        v = tl.load(v_ptr + b * svb + s * svs + h * D + jj, mask = mask, other = 0.0)
        tl.store(ov_ptr + out + jj, v, mask = mask)

    return kernel


def _launch_qk_rope(q, k, v, qw, kw, cos, sin, heads):
    """q / k / v: [B, S, H*D] bf16 (last dim contiguous); cos / sin: [1, S, 1, D]; returns [B, H, S, D] contiguous x3."""
    import triton

    batch, seq, width = q.shape
    dim = width // heads
    oq = q.new_empty((batch, heads, seq, dim))
    ok = torch_empty_like(oq)
    ov = torch_empty_like(oq)
    grid = (batch * seq, triton.cdiv(heads, _HEADS_PER_PROGRAM))
    _qk_rope_kernel()[grid](
        q,
        k,
        v,
        qw,
        kw,
        cos,
        sin,
        oq,
        ok,
        ov,
        seq,
        heads,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cos.stride(1),
        EPS = 1e-6,
        D = dim,
        HB = _HEADS_PER_PROGRAM,
        # ptxas otherwise FMA-fuses the rounded products into the sum on sm_100
        enable_fp_fusion = False,
    )
    return oq, ok, ov


def torch_empty_like(t):
    import torch
    return torch.empty_like(t)


def reference_qk_rope(q, k, v, qw, kw, cos, sin, heads):
    """The port's eager ops, in [B, H, S, D]."""
    import torch

    def norm(x, w):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim = True) + 1e-6)
        return y.to(x.dtype) * w

    def rot(x):
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat((-b, a), -1) * sin

    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    qh = rot(norm(q.unflatten(-1, (heads, -1)), qw).reshape(shape))
    kh = rot(norm(k.unflatten(-1, (heads, -1)), kw).reshape(shape))
    return qh.transpose(1, 2), kh.transpose(1, 2), v.reshape(shape).transpose(1, 2)


@functools.lru_cache(maxsize = 1)
def qk_rope_op() -> Any:
    """``unsloth_studio::moss_qk_norm_rope`` (opaque to Inductor, allocation only, CUDA-graph safe), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, "unsloth_studio", None)
    if ns is not None and hasattr(ns, _QK_ROPE_OP):
        return getattr(ns, _QK_ROPE_OP)
    if _qk_rope_kernel() is None:
        return None
    try:

        @torch.library.custom_op(
            f"unsloth_studio::{_QK_ROPE_OP}",
            mutates_args = (),
            schema = "(Tensor q, Tensor k, Tensor v, Tensor qw, Tensor kw, Tensor cos, Tensor sin, int heads)"
            " -> (Tensor, Tensor, Tensor)",
        )
        def _op(q, k, v, qw, kw, cos, sin, heads):
            return _launch_qk_rope(q, k, v, qw, kw, cos, sin, heads)

        @_op.register_fake
        def _(q, k, v, qw, kw, cos, sin, heads):
            batch, seq, width = q.shape
            out = q.new_empty((batch, heads, seq, width // heads))
            return out, torch.empty_like(out), torch.empty_like(out)
    except Exception:  # noqa: BLE001
        return None
    return getattr(torch.ops.unsloth_studio, _QK_ROPE_OP)


@functools.lru_cache(maxsize = 8)
def qk_rope_device_ok(index: int) -> bool:
    """The kernel runs and matches the port's eager ops on this device: every element within two bf16 ulps of its
    rotated pair's magnitude, and at
    most 0.5% of them off at all (the reduction order of the sum of squares), over ragged lengths and big values."""
    try:
        import torch

        op = qk_rope_op()
        if op is None:
            return False
        dev = torch.device("cuda", index)
        g = torch.Generator(device = "cpu").manual_seed(7)
        for seq, heads in ((37, 4), (1029, 24), (5, 3)):
            width = heads * 128
            qkv = (torch.randn(2, seq, 3 * width, generator = g) * 3).to(torch.bfloat16).to(dev)
            qkv[:, :, :7] *= 300
            q, k, v = qkv.chunk(3, -1)
            qw = (torch.rand(128, generator = g) + 0.5).to(torch.bfloat16).to(dev)
            kw = (torch.rand(128, generator = g) + 0.5).to(torch.bfloat16).to(dev)
            ang = torch.rand(seq, 64, generator = g) * 50
            cos = torch.cat((ang.cos(), ang.cos()), -1)[None, :, None].to(torch.bfloat16).to(dev)
            sin = torch.cat((ang.sin(), ang.sin()), -1)[None, :, None].to(torch.bfloat16).to(dev)
            got = op(q, k, v, qw, kw, cos, sin, heads)
            ref = reference_qk_rope(q, k, v, qw, kw, cos, sin, heads)
            if not torch.equal(got[2], ref[2]):
                return False
            for a, r in zip(got[:2], ref[:2]):
                diff = (a.float() - r.float()).abs()
                # A bf16 tie moves an output a few ulps of the pair magnitude.
                rf = r.float()
                pair = (
                    rf[..., : rf.shape[-1] // 2].square() + rf[..., rf.shape[-1] // 2 :].square()
                ).sqrt()
                ulp = (torch.cat((pair, pair), -1) * 2.0**-6).clamp_min(2.0**-14)
                if bool((diff > ulp).any()) or float((diff > 0).float().mean()) > 5e-3:
                    return False
        from .diffusion_cuda_graph import synchronize_outside_capture

        synchronize_outside_capture(dev)
        return True
    except Exception:  # noqa: BLE001
        return False
