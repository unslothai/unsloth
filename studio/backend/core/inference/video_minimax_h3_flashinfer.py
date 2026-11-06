# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 FlashInfer SM120 block kernels (``flashinfer.diffusion_ops``, FlashInfer >= 0.7.3), opt-in.

UNSLOTH_H3_FLASHINFER=1 swaps each resident FP8 / NVFP4 H3 block's forward for FlashInfer's fused SM120 operators:
pre-attention (norm1 + adaLN + QKV GEMM + q/k RMSNorm + RoPE), the out-projection with its gated residual and the MLP
(norm2 + adaLN + FC1 + SwiGLU + FC2 + gated residual). Attention stays on SDPA. Per-lever kill switches, each on by
default under the master switch: UNSLOTH_H3_FI_PRE, UNSLOTH_H3_FI_OUT, UNSLOTH_H3_FI_MLP.

Gate: CUDA sm_120 (the operators raise elsewhere), flashinfer >= 0.7.3 importable with the ops, H3 dimensions, a
resident denoiser, torchao Float8Tensor (per-row) or NVFP4Tensor weights. The operators JIT-compile on first use
(nvcc >= 12.8 and ninja; FlashInfer's jit-cache wheel does not carry them), hence opt-in. Weights are repacked from
Studio's own quantized tensors, bit-exact, into buffers on each block (they follow the module and die with it); the
replaced projections' own weights are then released (UNSLOTH_H3_FI_KEEP_STOCK=1 keeps them). NVFP4 pre-attention
needs one per-tensor scale for to_q / to_k / to_v (a prequant built with a shared QKV scale); other files keep the
stock pre-attention. A probe self-check against the stock block runs at install; any failure keeps the stock forward.

NVFP4 activation global scales: a1 / a2 / attention output from their amax per call (torchao's dynamic recipe);
the SwiGLU output y, which the fused MLP quantizes in-kernel, from its amax on the block's first call at each sequence
length, x UNSLOTH_H3_FI_NV_Y_MARGIN (default 4).
"""

from __future__ import annotations

import os
import threading
import types
from functools import lru_cache
from typing import Any, Optional

MASTER_ENV = "UNSLOTH_H3_FLASHINFER"
PRE_ENV = "UNSLOTH_H3_FI_PRE"
OUT_ENV = "UNSLOTH_H3_FI_OUT"
MLP_ENV = "UNSLOTH_H3_FI_MLP"
KEEP_STOCK_ENV = "UNSLOTH_H3_FI_KEEP_STOCK"
NV_Y_MARGIN_ENV = "UNSLOTH_H3_FI_NV_Y_MARGIN"

MIN_FLASHINFER = (0, 7, 3)
HIDDEN, HEADS, HEAD_DIM, FFN, ADALN_ROWS, ROPE_DIM = 5376, 56, 128, 14336, 9, 96
E2M1_X_E4M3 = 448.0 * 6.0
# Probe self-check bound on the relative L2 of the block's residual delta vs the stock block: a layout or scale bug
# lands near 1; FlashInfer vs stock on the same quantized weights measured 0.01-0.11 on real H3 inputs.
SELF_CHECK_TOL = {"fp8": 0.10, "nvfp4": 0.25}
_BUF = "_h3fi_"
_NS = "unsloth_h3fi"
_LOCK = threading.Lock()
# Host-side facts of each installed block's operands, keyed by the operand buffer's data pointer: the NVFP4 global
# weight scale (the kernels take alpha as a host float) and the calibrated y scale per sequence length. Ops read them
# inside the opaque op, so nothing per block reaches the traced graph: all 50 blocks share one compiled graph (the
# block buffers are graph inputs), with no recompile per block.
_HOST: dict = {}
_CAL: dict = {}
# The installed model's switches, the same for every block (read by the traced forward; constants for dynamo).
_CFG: dict = {
    "levers": (),
    "scheme": None,
    "cudnn": True,
    "released": False,
    "eps": 1e-5,
    "y_margin": 4.0,
}
# torch / F / SDPA handles for the traced forward, filled by register_ops(): no import statements in the region.
_T: dict = {}
CALLS: dict = {"pre": 0, "out": 0, "mlp": 0, "stock": 0, "y_calibrations": 0}


# Attention backends the fused block reproduces: unrestricted SDPA ("native" / unset) and cuDNN-pinned SDPA.
_SDPA_BACKENDS = ("", "native", "_native_cudnn")


def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip().lower()


def _on(name: str, default: str = "1") -> bool:
    return _env(name, default) not in ("0", "off", "false", "no", "")


def enabled() -> bool:
    return _on(MASTER_ENV, "0")


def _version_tuple(text: str) -> tuple:
    """Leading release numbers: "0.7.3" -> (0, 7, 3), "0.7.3rc1" -> (0, 7, 3), "0.7.3+cu130" -> (0, 7, 3)."""
    import re

    out = []
    for part in str(text).split("+")[0].split("."):
        m = re.match(r"\d+", part)
        if m is None:
            break
        out.append(int(m.group()))
        if m.end() != len(part):
            break
    return tuple(out)


def flashinfer_unavailable(device: Any = None) -> Optional[str]:
    """Why the FlashInfer H3 operators cannot run here, or None."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return f"torch unavailable: {exc}"
    from .rocm_bf16 import is_rocm_torch

    if not torch.cuda.is_available() or is_rocm_torch(torch):
        return "needs an NVIDIA CUDA device"
    try:
        cap = tuple(torch.cuda.get_device_capability(device))
    except Exception as exc:  # noqa: BLE001
        return f"no device capability: {exc}"
    if cap != (12, 0):
        return f"FlashInfer's MiniMax-H3 operators are sm_120 only (this device is sm_{cap[0]}{cap[1]})"
    try:
        import flashinfer
        from flashinfer import diffusion_ops as fd
    except Exception as exc:  # noqa: BLE001
        return f"flashinfer not importable: {type(exc).__name__}: {exc}"
    if _version_tuple(getattr(flashinfer, "__version__", "0")) < MIN_FLASHINFER:
        return f"flashinfer {getattr(flashinfer, '__version__', '?')} < 0.7.3"
    need = (
        "minimax_h3_fp8_pre_attention",
        "minimax_h3_nvfp4_pre_attention",
        "minimax_h3_fp8_out_proj",
        "minimax_h3_nvfp4_out_proj",
        "minimax_h3_mlp_fp8_sm120",
        "minimax_h3_mlp_nvfp4_sm120",
    )
    missing = [n for n in need if not hasattr(fd, n)]
    return f"flashinfer.diffusion_ops lacks {missing}" if missing else None


# ---------------------------------------------------------------------------------------------------------------- ops
def _host(t: Any) -> float:
    return _HOST[t.data_ptr()]


def _buf(block: Any, name: str) -> Any:
    return getattr(block, _BUF + name)


def _calibrated(key: tuple, amax_fn, margin: float, device: Any) -> tuple:
    hit = _CAL.get(key)
    if hit is None:
        import torch

        g = E2M1_X_E4M3 / max(float(amax_fn()) * margin, 1e-12)
        hit = (torch.tensor([g], dtype = torch.float32, device = device), g)
        with _LOCK:
            if len(_CAL) > 4096:
                _CAL.clear()
            _CAL[key] = hit
        CALLS["y_calibrations"] += 1
    return hit


@lru_cache(maxsize = None)
def register_ops() -> bool:
    import torch
    import torch.nn.functional as F
    from flashinfer import diffusion_ops as fd
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _T.update(
        {"torch": torch, "F": F, "cudnn": SDPBackend.CUDNN_ATTENTION, "sdpa_kernel": sdpa_kernel}
    )

    @torch.library.custom_op(
        f"{_NS}::pre",
        mutates_args = (),
        schema = "(Tensor x, Tensor scale9, Tensor shift9, Tensor idx, Tensor cs, Tensor ag, Tensor n1, "
        "Tensor qn, Tensor kn, Tensor wq, Tensor ws, float eps, int nv) -> (Tensor, Tensor, Tensor)",
    )
    def pre(x, scale9, shift9, idx, cs, ag, n1, qn, kn, wq, ws, eps, nv):
        CALLS["pre"] += 1
        if not nv:
            r = fd.minimax_h3_fp8_pre_attention(
                x, n1, scale9, shift9, idx, wq, ws, qn, kn, cs, eps = eps, out_mode = "bf16"
            )
        else:
            wg = _host(wq)
            r = fd.minimax_h3_nvfp4_pre_attention(
                x,
                n1,
                scale9,
                shift9,
                idx,
                wq,
                ws,
                wg,
                ag,
                qn,
                kn,
                cs,
                eps = eps,
                out_mode = "bf16",
                alpha = 1.0 / (float(ag.item()) * wg),
            )
        return r.q, r.k, r.v

    @pre.register_fake
    def _pre_fake(x, scale9, shift9, idx, cs, ag, n1, qn, kn, wq, ws, eps, nv):
        m = x.shape[0]
        return (
            x.new_empty((m, HEADS, HEAD_DIM)),
            x.new_empty((m, HEADS, HEAD_DIM)),
            x.new_empty((m, HEADS, HEAD_DIM)),
        )

    @torch.library.custom_op(
        f"{_NS}::out",
        mutates_args = (),
        schema = "(Tensor o, Tensor gate9, Tensor idx, Tensor residual, Tensor ag, Tensor wq, Tensor ws, "
        "int nv) -> Tensor",
    )
    def out(o, gate9, idx, residual, ag, wq, ws, nv):
        CALLS["out"] += 1
        if not nv:
            return fd.minimax_h3_fp8_out_proj(o, wq, ws, gate9, idx, residual)
        return fd.minimax_h3_nvfp4_out_proj(o, wq, ws, _host(wq), ag, gate9, idx, residual)

    @out.register_fake
    def _out_fake(o, gate9, idx, residual, ag, wq, ws, nv):
        return residual.new_empty(residual.shape)

    @torch.library.custom_op(
        f"{_NS}::mlp",
        mutates_args = (),
        schema = "(Tensor h, Tensor scale, Tensor shift, Tensor idx, Tensor gate, Tensor ag, Tensor n2, "
        "Tensor w1q, Tensor w1s, Tensor w2q, Tensor w2s, float eps, int nv) -> Tensor",
    )
    def mlp(h, scale, shift, idx, gate, ag, n2, w1q, w1s, w2q, w2s, eps, nv):
        CALLS["mlp"] += 1
        if not nv:
            return fd.minimax_h3_mlp_fp8_sm120(
                h, n2, scale, shift, idx, gate, h, w1q, w1s, w2q, w2s, eps = eps
            )
        ag_h = float(ag.item())
        g1, g2 = _host(w1q), _host(w2q)

        def y_amax():
            # y as the fused MLP forms it (FlashInfer's FC1 + SwiGLU on the same FC1 operands), once per length
            y = fd.minimax_h3_fc1_swiglu_nvfp4(
                h,
                n2,
                _pad_rows(scale),
                _pad_rows(shift),
                idx.int(),
                ag,
                w1q,
                w1s,
                torch.tensor([1.0 / (ag_h * g1)], dtype = torch.float32, device = h.device),
                eps = eps,
            )
            return y.abs().amax()

        yg_t, yg_h = _calibrated(
            (w1q.data_ptr(), int(h.shape[0])), y_amax, _CFG["y_margin"], h.device
        )
        return fd.minimax_h3_mlp_nvfp4_sm120(
            h,
            n2,
            scale,
            shift,
            idx,
            gate,
            h,
            ag,
            w1q,
            w1s,
            1.0 / (ag_h * g1),
            yg_t,
            w2q,
            w2s,
            1.0 / (yg_h * g2),
            eps = eps,
        )

    @mlp.register_fake
    def _mlp_fake(h, scale, shift, idx, gate, ag, n2, w1q, w1s, w2q, w2s, eps, nv):
        return h.new_empty(h.shape)

    return True


# ---------------------------------------------------------------------------------------------------------- weights
def _nvfp4_parts(weight: Any) -> tuple:
    """(codes uint8 [N, K / 2], row-major e4m3 block scales [N, K / 16], per-tensor scale float) of a torchao NVFP4Tensor."""
    import torch
    from torchao.prototype.mx_formats.utils import from_blocked

    n, k = weight.shape
    scale = weight.scale
    if getattr(weight, "is_swizzled_scales", False):
        scale = from_blocked(scale, n, k // 16)
    pts = weight.per_tensor_scale
    if pts is None:
        raise ValueError("NVFP4 weight without a per-tensor scale")
    return (
        weight.qdata.contiguous().view(torch.uint8).reshape(n, k // 2),
        scale.reshape(n, k // 16).contiguous(),
        float(pts.float().reshape(()).item()),
    )


def _nvfp4_dequant(codes: Any, scale: Any, pts: float) -> Any:
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4

    n = codes.shape[0]
    vals = f4_unpacked_to_f32(unpack_uint4(codes.contiguous())).view(
        n, -1, 16
    ) * scale.float().view(n, -1, 1)
    return vals.view(n, -1) * pts


def _swizzle(scale: Any) -> Any:
    """Row-major e4m3 [N, K / 16] -> FlashInfer's 128x4 swizzled flat uint8 (the cuBLAS block-scale layout)."""
    import torch
    from torchao.prototype.mx_formats.utils import to_blocked

    return to_blocked(scale.view(torch.float8_e4m3fn)).contiguous().view(torch.uint8).reshape(-1)


def _qkv_rows(t: Any) -> Any:
    """[q rows; k rows; v rows] -> FlashInfer's head / kind / dim row order."""
    import torch

    q, k, v = t.chunk(3, 0)
    tail = tuple(q.shape[1:])
    st = torch.stack(
        [
            q.reshape(HEADS, HEAD_DIM, *tail),
            k.reshape(HEADS, HEAD_DIM, *tail),
            v.reshape(HEADS, HEAD_DIM, *tail),
        ],
        1,
    )
    return st.reshape(3 * HEADS * HEAD_DIM, *tail).contiguous()


def _fc1_rows(t: Any) -> Any:
    """diffusers ``ff.net.0.proj`` rows [value; gate] -> the SM120 FC1 order: per 8 output columns, their 8 gate rows
    then their 8 value rows (FlashInfer's ``interleave_minimax_h3_fc1_rows_sm120`` of [gate; value])."""
    val, gate = t.chunk(2, 0)
    ffn = val.shape[0]
    tail = tuple(t.shape[1:])
    st = _stack_rows(gate.reshape(ffn // 8, 8, *tail), val.reshape(ffn // 8, 8, *tail))
    return st.reshape(2 * ffn, *tail).contiguous()


def _stack_rows(a: Any, b: Any) -> Any:
    import torch
    return torch.stack([a, b], 1)


def _scheme_of(weight: Any) -> Optional[str]:
    name = type(weight).__name__
    if name == "NVFP4Tensor":
        return "nvfp4"
    if name == "Float8Tensor":
        return "fp8"
    return None


def _linears(block: Any) -> dict:
    attn, ff = block.attn, block.ff
    return {
        "q": getattr(attn, "to_q", None),
        "k": getattr(attn, "to_k", None),
        "v": getattr(attn, "to_v", None),
        "o": attn.to_out[0],
        "fc1": ff.net[0].proj,
        "fc2": ff.net[2],
    }


def _pack(block: Any, scheme: str, levers: tuple) -> tuple:
    """Register FlashInfer's operands as non-persistent buffers on ``block``; returns (eps, levers it can take)."""
    import torch

    attn = block.attn
    lin = _linears(block)
    shapes = {
        "q": (HEADS * HEAD_DIM, HIDDEN),
        "k": (HEADS * HEAD_DIM, HIDDEN),
        "v": (HEADS * HEAD_DIM, HIDDEN),
        "o": (HIDDEN, HEADS * HEAD_DIM),
        "fc1": (2 * FFN, HIDDEN),
        "fc2": (HIDDEN, FFN),
    }
    for name, shape in shapes.items():
        m = lin[name]
        if (
            m is None
            or _scheme_of(getattr(m, "weight", None)) != scheme
            or getattr(m, "bias", None) is not None
        ):
            raise ValueError(f"{name}: not a bias-free {scheme} torchao Linear")
        if tuple(m.weight.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(m.weight.shape)} is not H3's {shape}")
    eps = {
        float(getattr(block.norm1, "eps", None) or 1e-5),
        float(getattr(block.norm2, "eps", None) or 1e-5),
        float(getattr(attn.norm_q, "eps", None) or 1e-5),
        float(getattr(attn.norm_k, "eps", None) or 1e-5),
    }
    if len(eps) != 1:
        raise ValueError(f"the fused kernels take one RMSNorm eps, the block has {sorted(eps)}")
    dev = lin["q"].weight.qdata.device

    def norm_w(m, n):
        w = getattr(m, "weight", None)
        return (
            (w if w is not None else torch.ones(n, device = dev))
            .detach()
            .to(dev, torch.bfloat16)
            .contiguous()
        )

    bufs = {
        "n1": norm_w(block.norm1, HIDDEN),
        "n2": norm_w(block.norm2, HIDDEN),
        "qn": norm_w(attn.norm_q, HEAD_DIM),
        "kn": norm_w(attn.norm_k, HEAD_DIM),
    }
    scales: dict = {}
    levers = list(levers)
    with torch.no_grad():
        if scheme == "fp8":
            parts = {n: (lin[n].weight.qdata, lin[n].weight.scale.float().reshape(-1)) for n in lin}
            for n, (q, sc) in parts.items():
                if q.dtype != torch.float8_e4m3fn or sc.numel() != q.shape[0]:
                    raise ValueError(f"{n}: not a per-row e4m3 Float8Tensor")
            if "pre" in levers:
                bufs["qkv_q"] = _qkv_rows(
                    torch.cat([parts[n][0].view(torch.uint8) for n in "qkv"], 0)
                ).view(torch.float8_e4m3fn)
                bufs["qkv_s"] = _qkv_rows(torch.cat([parts[n][1] for n in "qkv"], 0))
            if "out" in levers:
                bufs["o_q"], bufs["o_s"] = parts["o"][0].contiguous(), parts["o"][1].contiguous()
            if "mlp" in levers:
                bufs["fc1_q"] = _fc1_rows(parts["fc1"][0].view(torch.uint8)).view(
                    torch.float8_e4m3fn
                )
                bufs["fc1_s"] = _fc1_rows(parts["fc1"][1])
                bufs["fc2_q"], bufs["fc2_s"] = (
                    parts["fc2"][0].contiguous(),
                    parts["fc2"][1].contiguous(),
                )
        else:
            parts = {n: _nvfp4_parts(lin[n].weight) for n in lin}
            if "pre" in levers:
                pts = [parts[n][2] for n in "qkv"]
                if pts[0] == pts[1] == pts[2]:
                    codes = torch.cat([parts[n][0] for n in "qkv"], 0)
                    sc = torch.cat([parts[n][1].view(torch.uint8) for n in "qkv"], 0)
                    bufs["qkv_q"], bufs["qkv_s"] = _qkv_rows(codes), _swizzle(_qkv_rows(sc))
                    scales["qkv"] = 1.0 / pts[0]
                else:
                    # the fused QKV GEMM takes ONE global scale: re-scaling Studio's codes would requantize them
                    levers.remove("pre")

            def same_layout(name, sc):
                # the stored 128x4 swizzled scales ARE FlashInfer's layout when no row permutation applies: share them
                w = lin[name].weight
                if getattr(w, "is_swizzled_scales", False) and w.scale.is_contiguous():
                    return w.scale.view(torch.uint8).reshape(-1)
                return _swizzle(sc)

            if "out" in levers:
                c, sc, t = parts["o"]
                bufs["o_q"], bufs["o_s"], scales["o"] = c, same_layout("o", sc), 1.0 / t
            if "mlp" in levers:
                c, sc, t = parts["fc1"]
                bufs["fc1_q"], bufs["fc1_s"], scales["fc1"] = (
                    _fc1_rows(c),
                    _swizzle(_fc1_rows(sc.view(torch.uint8))),
                    1.0 / t,
                )
                c, sc, t = parts["fc2"]
                bufs["fc2_q"], bufs["fc2_s"], scales["fc2"] = c, same_layout("fc2", sc), 1.0 / t
    for name, t in bufs.items():
        block.register_buffer(_BUF + name, t, persistent = False)
    with _LOCK:
        for name, g in scales.items():
            _HOST[bufs[f"{name}_q"].data_ptr()] = g
    return eps.pop(), tuple(levers)


def _unpack(block: Any) -> None:
    for name in [n for n, _ in block.named_buffers(recurse = False) if n.startswith(_BUF)]:
        delattr(block, name)
    block.__dict__.pop("_unsloth_h3fi_stock", None)


def _release_stock(block: Any, levers: tuple) -> None:
    """Drop the stock weights the FlashInfer levers replaced: the block's buffers hold the same codes (the
    out-projection / FC2 buffers share the stock storage, QKV / FC1 are reordered copies)."""
    import torch

    lin = _linears(block)
    names = (
        (("q", "k", "v") if "pre" in levers else ())
        + (("o",) if "out" in levers else ())
        + (("fc1", "fc2") if "mlp" in levers else ())
    )
    for n in names:
        m = lin[n]
        m.weight = torch.nn.Parameter(
            torch.empty((0,), device = m.weight.device, dtype = torch.bfloat16), requires_grad = False
        )
        m._unsloth_h3fi_released = True


# ------------------------------------------------------------------------------------------------------ block forward
def _pad_rows(t: Any) -> Any:
    """[rows, C] -> the [9, C] table the fused kernels index (no branch on rows: it is unbacked under compile)."""
    return _T["torch"].cat([t, t.new_zeros((ADALN_ROWS - t.shape[0], t.shape[1]))], 0)


def _ag(t: Any) -> Any:
    return (E2M1_X_E4M3 / t.abs().amax().float().clamp_min(1e-12)).reshape(1)


def _rope(x: Any, cos: Any, sin: Any) -> Any:
    """diffusers' ``_apply_rotary_emb`` on [M, H, D] (rotate-half over the first ``cos.shape[-1]`` channels)."""
    torch = _T["torch"]
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    c, s = cos.to(x.dtype)[:, None, :], sin.to(x.dtype)[:, None, :]
    x1, x2 = xr.chunk(2, dim = -1)
    return torch.cat((xr * c + torch.cat((-x2, x1), dim = -1) * s, xp), dim = -1)


def _fi_one(self: Any, x: Any, mods: tuple, idx64: Any, cos: Any, sin: Any) -> Any:
    """One packed sequence [M, C] through the block. Everything per block is a module tensor (a graph input), the
    switches are _CFG's model-wide constants: one compiled graph serves all blocks."""
    torch, F = _T["torch"], _T["F"]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
    levers, nv, eps = _CFG["levers"], int(_CFG["scheme"] == "nvfp4"), _CFG["eps"]
    idx32 = idx64.int()
    empty = x.new_empty((1,), dtype = torch.float32)
    attn = self.attn
    if "pre" in levers:
        cs = torch.cat([cos[:, : ROPE_DIM // 2], sin[:, : ROPE_DIM // 2]], -1).to(torch.bfloat16)
        ag1 = empty
        if nv:
            n = F.rms_norm(x, (x.shape[-1],), self.norm1.weight, self.norm1.eps)
            ag1 = _ag(
                n * (1.0 + scale_msa.index_select(0, idx64)) + shift_msa.index_select(0, idx64)
            )
        q, k, v = torch.ops.unsloth_h3fi.pre(
            x,
            _pad_rows(scale_msa),
            _pad_rows(shift_msa),
            idx32,
            cs,
            ag1,
            self._h3fi_n1,
            self._h3fi_qn,
            self._h3fi_kn,
            self._h3fi_qkv_q,
            self._h3fi_qkv_s,
            eps,
            nv,
        )
    else:
        n = self.norm1(x)
        am = n * (1.0 + scale_msa.index_select(0, idx64)) + shift_msa.index_select(0, idx64)
        q = attn.norm_q(attn.to_q(am).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(am).unflatten(-1, (attn.heads, -1)))
        v = attn.to_v(am).unflatten(-1, (attn.heads, -1))
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    qp, kp, vp = (t[None].permute(0, 2, 1, 3) for t in (q, k, v))
    if _CFG["cudnn"]:
        with _T["sdpa_kernel"](_T["cudnn"]):
            o = F.scaled_dot_product_attention(qp, kp, vp, dropout_p = 0.0, is_causal = False)
    else:
        o = F.scaled_dot_product_attention(qp, kp, vp, dropout_p = 0.0, is_causal = False)
    o = o.permute(0, 2, 1, 3)[0].reshape(x.shape[0], -1).to(torch.bfloat16)
    if "out" in levers:
        h = torch.ops.unsloth_h3fi.out(
            o,
            _pad_rows(gate_msa),
            idx32,
            x,
            _ag(o) if nv else empty,
            self._h3fi_o_q,
            self._h3fi_o_s,
            nv,
        )
    else:
        h = x + gate_msa.index_select(0, idx64) * attn.to_out[0](o)
    if "mlp" in levers:
        ag2 = empty
        if nv:
            n = F.rms_norm(h, (h.shape[-1],), self.norm2.weight, self.norm2.eps)
            ag2 = _ag(
                n * (1.0 + scale_mlp.index_select(0, idx64)) + shift_mlp.index_select(0, idx64)
            )
        return torch.ops.unsloth_h3fi.mlp(
            h,
            scale_mlp.contiguous(),
            shift_mlp.contiguous(),
            idx64,
            gate_mlp.contiguous(),
            ag2,
            self._h3fi_n2,
            self._h3fi_fc1_q,
            self._h3fi_fc1_s,
            self._h3fi_fc2_q,
            self._h3fi_fc2_s,
            eps,
            nv,
        )
    n = self.norm2(h)
    am = n * (1.0 + scale_mlp.index_select(0, idx64)) + shift_mlp.index_select(0, idx64)
    return h + gate_mlp.index_select(0, idx64) * self.ff(am)


def _fi_block_forward(
    self: Any,
    hidden_states: Any,
    temb: Any,
    adaln_indices: Any,
    rotary_emb: Any,
    attention_mask: Any = None,
) -> Any:
    torch = _T["torch"]
    mods = self.adaln_proj(temb)
    if (
        attention_mask is not None
        or hidden_states.dim() != 3
        or hidden_states.dtype != torch.bfloat16
        or (not torch.compiler.is_compiling() and mods[0].shape[0] > ADALN_ROWS)
    ):
        if _CFG["released"]:
            raise RuntimeError(
                "MiniMax-H3 FlashInfer blocks (stock weights released) take no attention mask, bf16 "
                "only, at most 9 adaLN rows; set UNSLOTH_H3_FLASHINFER=0"
            )
        CALLS["stock"] += 1
        return self._unsloth_h3fi_stock(
            hidden_states, temb, adaln_indices, rotary_emb, attention_mask
        )
    # the adaLN row count (3 x distinct timesteps) is data-dependent under compile: no guard on it there, only a check
    torch._check(
        mods[0].shape[0] <= ADALN_ROWS,
        lambda: "MiniMax-H3 FlashInfer blocks take at most 9 adaLN rows",
    )
    idx64 = adaln_indices.reshape(-1).long()
    cos, sin = rotary_emb
    outs = [
        _fi_one(self, hidden_states[b], mods, idx64, cos, sin)
        for b in range(hidden_states.shape[0])
    ]
    return outs[0][None] if len(outs) == 1 else torch.stack(outs, 0)


# ------------------------------------------------------------------------------------------------------------ install
def _probe(block: Any, m: int = 384) -> tuple:
    import torch

    dev = next(block.parameters()).device
    g = torch.Generator(device = "cpu").manual_seed(1234)
    x = torch.randn(1, m, HIDDEN, generator = g).to(dev, torch.bfloat16)
    lin = block.adaln_proj.linear
    temb = (torch.randn(1, lin.in_features, generator = g) * 0.5).to(dev, lin.weight.dtype)
    idx = (torch.arange(m) % 3).to(dev)
    pos = torch.stack(
        [torch.arange(m) // 64, (torch.arange(m) // 8) % 8, torch.arange(m) % 8], -1
    ).float()
    inv = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype = torch.float32) / 32))
    fr = torch.cat([pos[:, i : i + 1] * inv[None] for i in range(3)], -1)
    fr = torch.cat([fr, fr], -1).to(dev)
    return x, temb, idx, (fr.cos(), fr.sin())


def _self_check(block: Any) -> float:
    import torch

    x, temb, idx, rot = _probe(block)
    with torch.no_grad():
        ref = block._unsloth_h3fi_stock(x, temb, idx, rot, None).float() - x.float()
        got = _fi_block_forward(block, x, temb, idx, rot, None).float() - x.float()
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    _CAL.clear()  # the probe must not leave its calibration behind
    return rel if rel == rel else float("inf")


def _qkv_shared(block: Any) -> bool:
    lin = _linears(block)
    pts = [getattr(lin[n].weight, "per_tensor_scale", None) for n in "qkv"]
    if any(p is None for p in pts):
        return False
    return bool(torch_equal(pts[0], pts[1]) and torch_equal(pts[0], pts[2]))


def torch_equal(a: Any, b: Any) -> bool:
    import torch
    return bool(torch.equal(a.float().reshape(-1), b.float().reshape(-1)))


_TOOLCHAIN_HINT = (
    "first use JIT-compiles these kernels: needs nvcc >= 12.8 (PATH or CUDA_HOME) and ninja"
)


def _failure_reason(exc: BaseException, limit: int = 400) -> str:
    """One log line for an install failure. A failed JIT build's message starts with ninja echoing whole nvcc command
    lines; keep its diagnosis (e.g. "nvcc: not found", the compiler error) and name the toolchain it needs."""
    text = str(exc)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    toolchain = "Ninja build failed" in text or "nvcc" in text or "ninja" in text
    if "Ninja build failed" in text:
        lines = [
            ln
            for ln in lines
            if len(ln) <= 300 and not ln.startswith(("[", "ninja: Entering", "Ninja build failed"))
        ] or lines[:1]
        lines = ["JIT build failed"] + lines
    msg = f"{type(exc).__name__}: " + " | ".join(lines)
    if toolchain:
        msg += f" ({_TOOLCHAIN_HINT})"
    if len(msg) > limit:
        msg = msg[: limit // 4] + " ... " + msg[-(limit - limit // 4 - 5) :]
    return msg


def install_h3_flashinfer(
    transformer: Any,
    *,
    resident: bool = True,
    logger: Any = None,
) -> dict:
    """Swap the resident H3 blocks' forward for the FlashInfer SM120 operators; returns what engaged."""
    result: dict = {"blocks": 0, "scheme": None, "levers": (), "reason": None}
    if not enabled():
        return result
    blocks = list(getattr(transformer, "transformer_blocks", None) or [])
    if not blocks:
        result["reason"] = "no transformer_blocks"
        return result
    first = blocks[0]
    try:
        device = next(first.parameters()).device
    except Exception:  # noqa: BLE001
        device = None
    # the pure-Python gates first: a streamed or int8 / bf16 denoiser never imports flashinfer
    reason = None
    if not resident:
        reason = "the denoiser is streamed / offloaded (the operators need it resident)"
    q_lin = getattr(
        getattr(first, "attn", None), "to_q", None
    )  # absent when Studio fused the int8 QKV
    scheme = _scheme_of(getattr(q_lin, "weight", None))
    if reason is None and scheme is None:
        reason = (
            f"blocks are not torchao fp8 / nvfp4 ({type(getattr(q_lin, 'weight', None)).__name__})"
        )
    levers = tuple(
        n for n, env in (("pre", PRE_ENV), ("out", OUT_ENV), ("mlp", MLP_ENV)) if _on(env)
    )
    if reason is None and not levers:
        reason = "every lever switched off"
    if reason is None:
        reason = flashinfer_unavailable(device)
    if reason is not None:
        result["reason"] = reason
        if logger is not None:
            logger.info("video.h3_flashinfer: off: %s", reason)
        return result
    register_ops()
    proc = getattr(first.attn, "processor", None)
    backend = str(
        getattr(
            getattr(proc, "_attention_backend", None),
            "value",
            getattr(proc, "_attention_backend", ""),
        )
        or ""
    )
    if backend not in _SDPA_BACKENDS:
        # the fused block calls SDPA itself (cuDNN-pinned or unrestricted); any other selected backend would be overridden
        result["reason"] = f"attention backend {backend!r} is not reproduced by the fused block"
        if logger is not None:
            logger.info("video.h3_flashinfer: off: %s", result["reason"])
        return result
    try:
        margin = float(_env(NV_Y_MARGIN_ENV, "4"))
    except ValueError:
        margin = 4.0
    # 0 / negative / nan / inf would turn the y scale into garbage
    if not 0.0 < margin < float("inf"):
        margin = 4.0
    if scheme == "nvfp4" and "pre" in levers and not all(_qkv_shared(b) for b in blocks):
        # the fused QKV GEMM takes ONE global scale: re-scaling Studio's per-projection codes would requantize them
        levers = tuple(n for n in levers if n != "pre")
        if logger is not None:
            logger.info(
                "video.h3_flashinfer: q / k / v do not share one NVFP4 scale: stock pre-attention"
            )
        if not levers:
            result["reason"] = (
                "only the pre-attention lever was on and the NVFP4 q / k / v scales differ"
            )
            return result
    _CFG.update(
        {
            "levers": levers,
            "scheme": scheme,
            "cudnn": backend == "_native_cudnn",
            "released": False,
            "y_margin": margin,
        }
    )
    import torch

    def pack(blk: Any) -> float:
        eps, took = _pack(blk, scheme, levers)
        if took != levers:
            _unpack(blk)
            raise ValueError(f"block took {took}, not {levers}")
        blk._unsloth_h3fi_stock = blk.forward
        return eps

    try:
        _CFG["eps"] = pack(first)
        rel = _self_check(first)
        if not rel <= SELF_CHECK_TOL[scheme]:
            raise RuntimeError(
                f"self-check: block delta rel L2 {rel:.4f} > {SELF_CHECK_TOL[scheme]}"
            )
    except Exception as exc:  # noqa: BLE001 -- the stock forward stays
        _unpack(first)
        _CFG["levers"] = ()
        result["reason"] = _failure_reason(exc)
        if logger is not None:
            logger.warning("video.h3_flashinfer: kept the stock blocks: %s", result["reason"])
        return result
    keep = _on(KEEP_STOCK_ENV, "0")
    _CFG["released"] = not keep
    on_cuda = device is not None and device.type == "cuda"
    freed = 0
    installed: list = []
    # one block at a time: pack, switch, release, so the load-time peak holds one block's copies, not all of them
    for blk in blocks:
        try:
            if blk is not first and pack(blk) != _CFG["eps"]:
                _unpack(blk)
                raise ValueError("blocks disagree on the RMSNorm eps")
        except Exception as exc:  # noqa: BLE001 -- this block and the rest stay stock
            result["reason"] = f"block {len(installed)}: {type(exc).__name__}: {exc}"[:400]
            if logger is not None:
                logger.warning(
                    "video.h3_flashinfer: blocks from %d on kept stock: %s",
                    len(installed),
                    result["reason"],
                )
            break
        blk.forward = types.MethodType(_fi_block_forward, blk)
        if not keep:
            before = torch.cuda.memory_allocated(device) if on_cuda else 0
            _release_stock(blk, levers)
            freed += before - (torch.cuda.memory_allocated(device) if on_cuda else 0)
        installed.append(blk)
    if on_cuda and not keep:
        # hand the released weights' allocator segments back to the driver (-3.5 GiB render peak at 960x544x124)
        torch.cuda.empty_cache()
    engaged = set(levers)
    result.update(
        {
            "blocks": len(installed),
            "scheme": scheme,
            "levers": tuple(sorted(engaged)),
            "self_check": rel,
            "released_bytes": freed,
        }
    )
    if logger is not None:
        logger.info(
            "video.h3_flashinfer: %d %s blocks on FlashInfer (%s), self-check rel %.4f, %.2f GB of stock "
            "weights released",
            len(installed),
            scheme,
            "+".join(sorted(engaged)),
            rel,
            freed / 1e9,
        )
    return result
