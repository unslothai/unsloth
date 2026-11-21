# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DoRA's per-row weight norm without building the dense B @ A update.

PEFT's DoraLinearLayer.forward rebuilds the dense update at every call: torch.eye(d_in) pushed through lora_A and
lora_B, then W + s * (B @ A) and its row norm. That is several weight-sized passes and temporaries per DoRA module
per forward, twice per step under gradient checkpointing. The norm only needs rank-sized work
(arXiv 2603.22276, "Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels"):

    ||W_i + s B_i A||^2 = ||W_i||^2 + 2 s (W A^T)_i . B_i + s^2 B_i (A A^T) B_i^T

so one d_out x d_in x r GEMM and one row reduction of W replace the dense update. PEFT detaches the norm (DoRA
paper section 4.3), so no gradient changes. The cross term is a float32 GEMM output and the rank-sized terms are
float64, so the norm is closer to an fp64 reference than PEFT's W + s * (B @ A) rounded to the weight dtype element
by element. The norm reads the 16-bit (or dequantized) base weight as stored, never a float32 copy of it. The rest
of the forward (magnitude / norm scale, bias, dropout path) is PEFT's own, line for line.

Anything this does not cover runs PEFT's original forward: float32 / float64 base weights, subclasses (embedding, conv),
PEFT's eval-time DoRA cache, DTensor / sharded weights, weights on another device (ephemeral offload), non-2D or
mismatched shapes. UNSLOTH_DORA_FACTORED_NORM=0 turns it off.
"""

import functools
import os

import torch
import torch.nn.functional as F

__all__ = ["STATS", "patch_peft_dora_factored_norm", "factored_dora_weight_norm"]

STATS = {"factored": 0, "fallback": 0}


def _count(key):
    # Eager only: a traced += on a module-level int guards on its value, so every call would recompile
    # (Unsloth's per-block compile then gives up and runs the block eagerly).
    if not torch.compiler.is_compiling():
        STATS[key] += 1


def _plain_2d(t):
    # DTensor and other wrapper subclasses keep PEFT's path; nn.Parameter is a plain tensor here.
    return (
        isinstance(t, torch.Tensor)
        and type(t) in (torch.Tensor, torch.nn.Parameter)
        and t.ndim == 2
    )


_HALF = (torch.float16, torch.bfloat16)
# aten::mm.dtype (16-bit inputs, float32 output) exists on CUDA in recent torch only.
_MM_OUT_DTYPE = hasattr(torch.ops.aten.mm, "dtype") and getattr(torch.version, "hip", None) is None
_CHUNK_BYTES = 64 * 1024 * 1024


# Largest rank whose float64 rank terms stay cheap on GPUs with 1/64-rate float64 (T4, L4, RTX): 4 r^2 (d_in + d_out)
# float64 FLOPs per call, 34 us on an L4 at r = 32 and d = 4096, but 2.2 ms at r = 256 (an RTX 3090 rank-256 DoRA step
# went from 95 to 261 ms).
_FP64_MAX_RANK_GPU = 32


def _rank_dtype(device, rank = 0):
    # Rank-sized terms in float64 on CPU, and on CUDA / ROCm up to _FP64_MAX_RANK_GPU: a float32 matmul may run as TF32
    # (only if the user enabled it) and round A / B to 10 bits. Other devices may have no float64 (MPS; Intel Arc XPUs
    # raise "Required aspect fp64"): float32. rank is a static weight shape, so the choice adds no runtime guard.
    if device.type == "cpu":
        return torch.float64
    if device.type == "cuda" and rank <= _FP64_MAX_RANK_GPU:
        return torch.float64
    return torch.float32


def _cross_term_fp32(weight, lora_A_weight):
    """(W A^T)^T as (r, d_out) float32 with float32 accumulation, A at its stored precision.

    A float32 A is split into two 16-bit parts (hi + lo) stacked into one (2r, d_in) operand, so W is read once and
    every product is exact; the GEMM writes float32, never a rounded 16-bit cross term.
    """
    lowp = weight.dtype
    r = lora_A_weight.shape[0]
    A_hi = lora_A_weight.to(lowp)
    split = lora_A_weight.dtype != lowp
    if split:
        A_lo = (lora_A_weight.float() - A_hi.float()).to(lowp)
        A_2 = torch.cat((A_hi, A_lo))
    else:
        A_2 = A_hi
    if _MM_OUT_DTYPE and weight.is_cuda and not (lowp == torch.float16 and _fp16_accumulation()):
        WA = torch.mm(A_2, weight.t(), out_dtype = torch.float32)
    else:
        # 16-bit values are exact in float32 (and in TF32); upcast W in bounded row chunks.
        A_f = A_2.float()
        rows = max(1, _CHUNK_BYTES // (4 * weight.shape[1]))
        WA = torch.cat(
            [F.linear(A_f, weight[i : i + rows].float()) for i in range(0, weight.shape[0], rows)],
            dim = 1,
        )
    return WA[:r] + WA[r:] if split else WA


def _fp16_accumulation():
    # cuBLAS refuses float16 inputs with a float32 output under allow_fp16_accumulation (CUDABlas.cpp,
    # gemm_internal<at::Half, float>). Reading the flag is a graph break, so a trace takes it as a constant.
    return bool(getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False))


_assume_constant = getattr(torch.compiler, "assume_constant_result", None)
if _assume_constant is not None:
    _fp16_accumulation = _assume_constant(_fp16_accumulation)


@torch.no_grad()
def factored_dora_weight_norm(weight, lora_A_weight, lora_B_weight, scaling, dtype):
    """Row norm of weight + scaling * lora_B_weight @ lora_A_weight, without forming the product.

    weight: (d_out, d_in), float16 / bfloat16. lora_A_weight: (r, d_in). lora_B_weight: (d_out, r).
    Returns (d_out,) in dtype. The squared norm is a difference of large terms when s B A nearly cancels W, so
    the cross term is kept in float32 (exact products, float32 accumulation), the rank-sized terms in float64,
    autocast is off, and the squared norm is floored at the float32 rounding level of its terms instead of 0.
    """
    with torch.autocast(device_type = weight.device.type, enabled = False):
        acc = _rank_dtype(weight.device, lora_A_weight.shape[0])
        A = lora_A_weight.to(acc)
        B = lora_B_weight.to(acc)
        WA = _cross_term_fp32(weight, lora_A_weight).to(acc)
        base = torch.linalg.vector_norm(weight, ord = 2, dim = 1, dtype = torch.float32).to(acc).square_()
        cross = (WA.t() * B).sum(1).mul_(2.0 * scaling)
        quad = ((B @ (A @ A.t())) * B).sum(1).mul_(scaling * scaling)
        norm_sq = base + cross + quad
        floor = (base + cross.abs() + quad).mul_(torch.finfo(torch.float32).eps)
        return torch.maximum(norm_sq, floor).sqrt_().to(dtype)


def patch_peft_dora_factored_norm():
    if os.environ.get("UNSLOTH_DORA_FACTORED_NORM", "1") == "0":
        return False
    try:
        from peft.tuners.lora import dora as peft_dora
    except Exception:
        return False
    cls = getattr(peft_dora, "DoraLinearLayer", None)
    forward = getattr(cls, "forward", None)
    if forward is None:
        return False
    if getattr(forward, "_unsloth_factored_norm", False):
        return True

    @functools.wraps(forward)
    def dora_forward(
        self,
        x,
        *,
        lora_A,
        lora_B,
        scaling,
        base_layer,
        base_result = None,
        **kwargs,
    ):
        lora_A_weight = getattr(lora_A, "weight", None)
        lora_B_weight = getattr(lora_B, "weight", None)
        weight = None
        if (
            type(self) is cls
            and not (getattr(peft_dora, "ENABLE_DORA_CACHING", False) and not self.training)
            and _plain_2d(lora_A_weight)
            and _plain_2d(lora_B_weight)
            and lora_A_weight.device == x.device
            and lora_B_weight.device == x.device
        ):
            weight = peft_dora.dequantize_module_weight(base_layer)
            # 16-bit (or dequantized to 16-bit) base weights only: a float32 base keeps PEFT's float32 dense sum.
            if _plain_2d(weight) and weight.dtype in _HALF:
                weight = peft_dora.transpose(weight, self.fan_in_fan_out)
                if x.dtype in _HALF:
                    weight = weight.to(x.dtype)
                if weight.device != x.device or weight.shape != (
                    lora_B_weight.shape[0],
                    lora_A_weight.shape[1],
                ):
                    weight = None
            else:
                weight = None
        if weight is None:
            _count("fallback")
            return forward(
                self,
                x,
                lora_A = lora_A,
                lora_B = lora_B,
                scaling = scaling,
                base_layer = base_layer,
                base_result = base_result,
                **kwargs,
            )
        _count("factored")
        magnitude = self.weight
        weight_norm = factored_dora_weight_norm(
            weight, lora_A_weight, lora_B_weight, scaling, x.dtype
        )
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        lora_result = lora_B(lora_A(x))

        bias = None
        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                base_result = base_result - bias
        else:
            base_result = F.linear(x, weight.to(x.dtype))

        return (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling

    dora_forward._unsloth_factored_norm = True
    dora_forward._unsloth_original = forward
    cls.forward = dora_forward
    return True
