# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes NF4 ``Linear4bit.forward`` on Unsloth's NF4 kernels: Triton dequantize + F.linear
(bit-exact to bitsandbytes' dequantize) and ``fast_gemv`` for one decode row; dX recomputes the
weight as bitsandbytes does. From bitsandbytes 0.50, shapes its fused 4-bit GEMM takes stay with it.
Default on before bitsandbytes 0.50, and from 0.50 only on sm100 / sm120 (A100 / T4 batched decode
measured 2-3% slower). UNSLOTH_BNB_NF4_LINEAR=1/0 forces it; UNSLOTH_BNB_TRITON=0 disables it."""

import os

import torch
import torch.nn.functional as F

from . import utils as _U

__all__ = ["install_bnb_nf4_override", "uninstall_bnb_nf4_override"]

_ENABLED = os.environ.get("UNSLOTH_BNB_TRITON", "1") != "0"
_LINEAR = os.environ.get("UNSLOTH_BNB_NF4_LINEAR", "auto")
_MARK = "_unsloth_nf4_override"
_PICKS = "_unsloth_nf4_bnb_fused"  # per-layer {x.numel(): bitsandbytes runs its fused GEMM}
_FALLBACK = {}

try:
    from unsloth_zoo.utils import Version

    # Before 2.11 Inductor miscompiles the traced backward of these Functions: run opaque.
    _TRACE = Version(torch.__version__) >= Version("2.11.0") and hasattr(torch.library, "triton_op")
except Exception:
    _TRACE = False


def _bf16_traceable():
    # bf16 user Triton cannot be re-emitted on sm75 (T4); any visible GPU may hold a layer.
    try:
        if torch.version.hip:
            return True
        n = torch.cuda.device_count()
        return n > 0 and all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(n))
    except Exception:
        return False


_TRACE_BF16 = _TRACE and _bf16_traceable()


def _apply(function, *args):
    # Dynamo special-cases Function.apply by name, so route through a plain function.
    return function.apply(*args)


class _NF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, quant_state, bias):
        ctx.unsloth_nf4 = (weight, quant_state)
        return F.linear(x, _U.fast_dequantize(weight, quant_state), bias)

    @staticmethod
    def backward(ctx, dY):
        weight, quant_state = ctx.unsloth_nf4
        W = _U.fast_dequantize(weight, quant_state)
        return torch.matmul(dY, W.to(dY.dtype)), None, None, None


def _linear(x, weight, quant_state, bias):
    if x.requires_grad and torch.is_grad_enabled():
        return _apply(_NF4Linear, x, weight, quant_state, bias)
    K = x.shape[-1]
    if x.numel() == K and quant_state.dtype is not torch.float32:
        N = quant_state.shape[0]
        out = _U.fast_gemv(x.reshape(1, 1, K), weight, quant_state)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], N)
    return F.linear(x, _U.fast_dequantize(weight, quant_state), bias)


def _bnb_ops_traceable():
    # bitsandbytes >= 0.46 uses registered torch.library ops (traceable); 0.45.5 calls ctypes.
    try:
        return hasattr(torch.ops.bitsandbytes, "dequantize_4bit")
    except Exception:
        return False


_BNB_OPS = False
# bitsandbytes >= 0.50 fused 4-bit GEMM row cap (_gemm_4bit_custom_max_m).
_BNB_FUSED = False
_BNB_FUSED_MAX_ROWS = 1536
_BNB_PICK = None


def _bnb_has_fused_gemm():
    try:
        return hasattr(torch.ops.bitsandbytes, "gemm_4bit")
    except Exception:
        return False


def _bnb_fused_heuristic():
    # Mirrors bitsandbytes' own fused-vs-dequantize choice for gemm_4bit (0.50).
    try:
        from bitsandbytes.backends.cuda import ops

        pick = ops._gemm_4bit_use_custom_fn
        max_rows = int(ops._gemm_4bit_custom_max_m)
        return (pick, max_rows) if callable(pick) else (None, 1536)
    except Exception:
        return None, 1536


def _default_on(capabilities, has_fused_gemm, hip):
    """On where measured faster with no regressed shape: every CUDA GPU sm75+ before bitsandbytes
    0.50; from 0.50 (its fused small-batch GEMM) only when every GPU is sm100 or sm120. Off on ROCm."""
    if hip or not capabilities or not all(cap >= (7, 5) for cap in capabilities):
        return False
    if not has_fused_gemm:
        return True
    return all(major in (10, 12) for major, _ in capabilities)


def _wanted():
    if _LINEAR == "0":
        return False
    if _LINEAR == "1":
        return True
    try:
        caps = [
            tuple(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return False
    return _default_on(caps, _bnb_has_fused_gemm(), torch.version.hip is not None)


def _eligible(self, x, weight, quant_state):
    """Every condition under which the replacement equals the forward it replaces."""
    if (
        quant_state is None
        or type(quant_state) is list
        or x.device.type != "cuda"
        or torch.version.hip is not None
        or weight.dtype is not torch.uint8
        or x.numel() == 0
        or not _U._USE_NF4_KERNELS
        or getattr(quant_state, "quant_type", None) != "nf4"
        or getattr(quant_state, "packing_format_for_cpu", False)
    ):
        return False
    shape = quant_state.shape
    dtype = quant_state.dtype
    blocksize = quant_state.blocksize
    if (
        len(shape) != 2
        or dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or x.shape[-1] != shape[1]
        or shape[1] % blocksize != 0
        or blocksize & (blocksize - 1) != 0
        or weight.numel() * 2 != shape[0] * shape[1]
    ):
        return False
    state2 = quant_state.state2
    if quant_state.nested and (state2 is None or state2.blocksize != 256):
        # bitsandbytes raises for any other nested blocksize; keep its behaviour.
        return False
    # Dequantizing straight to the compute dtype equals bitsandbytes' only when the dtypes match.
    if self.compute_dtype is not dtype:
        return False
    bias = self.bias
    if bias is not None and bias.requires_grad and torch.is_grad_enabled():
        return False
    if (
        torch.is_autocast_enabled(x.device.type)
        and torch.get_autocast_dtype(x.device.type) is not dtype
    ):
        return False
    return True


def _bnb_fused_takes(self, x, quant_state):
    """bitsandbytes >= 0.50 would run its fused 4-bit GEMM here (2..1536 rows, per its own per-GPU
    heuristic): keep that. Where it dequantizes + F.linear instead, the override does the same with
    the faster Triton dequantize, bit for bit. Eager answers are cached per layer and element count
    (_nf4_forward checks that cache first): this runs on every decode call."""
    K = x.shape[-1]
    n = x.numel()
    if n <= K or n > _BNB_FUSED_MAX_ROWS * K:
        return False
    if _BNB_PICK is None or torch.compiler.is_compiling():
        # No heuristic to ask, or traced (symbolic sizes): leave the whole range to bitsandbytes.
        return True
    picks = self.__dict__.get(_PICKS)
    take = None if picks is None else picks.get(n)
    if take is not None:
        return take
    try:
        take = bool(_BNB_PICK(x.device.index, quant_state.dtype, n // K, quant_state.shape[0], K))
    except Exception:
        take = True
    if picks is None or len(picks) > 4096:
        picks = self.__dict__[_PICKS] = {}
    picks[n] = take
    return take


def _forward(self, x):
    weight = self.weight
    quant_state = getattr(weight, "quant_state", None)
    if (
        _BNB_FUSED
        and quant_state is not None
        and type(quant_state) is not list
        and _bnb_fused_takes(self, x, quant_state)
    ):
        return _FALLBACK["forward"](self, x)
    if not self.compute_type_is_set or not _eligible(self, x, weight, quant_state):
        return _FALLBACK["forward"](self, x)
    dtype = quant_state.dtype
    if dtype is torch.float32 and x.numel() < 8 * x.shape[-1]:
        # bitsandbytes >= 0.50 runs fp32 below 8 rows on its own, more accurate kernel.
        return _FALLBACK["forward"](self, x)
    if dtype is torch.bfloat16 and not _TRACE_BF16 and torch.compiler.is_compiling():
        # sm75 cannot re-emit bf16 user Triton: keep bitsandbytes' (traceable) forward.
        if _BNB_OPS:
            return _FALLBACK["forward"](self, x)
        return _forward_opaque(self, x)
    inp_dtype = x.dtype
    x = x.to(dtype)
    bias = self.bias
    if bias is not None:
        bias = bias.to(dtype)
    return _linear(x, weight, quant_state, bias).to(inp_dtype)


_forward_opaque = torch._dynamo.disable(_forward)


def _nf4_forward(self, x: torch.Tensor):
    if _BNB_FUSED and not torch.compiler.is_compiling():
        picks = self.__dict__.get(_PICKS)
        if picks is not None and picks.get(x.numel()) is True:
            return _FALLBACK["forward"](self, x)
    if not _TRACE and torch.compiler.is_compiling():
        # Before torch 2.11: keep traceable bitsandbytes (>= 0.46), else one opaque call.
        if _BNB_OPS:
            return _FALLBACK["forward"](self, x)
        return _forward_opaque(self, x)
    return _forward(self, x)


_nf4_forward._unsloth_nf4_override = True


def install_bnb_nf4_override():
    """Route NF4 bitsandbytes Linear4bit forwards through Unsloth's NF4 kernels. Idempotent; the
    forward installed before (bitsandbytes' or zoo's patched one) handles everything else."""
    if not _ENABLED:
        return False
    try:
        import bitsandbytes as bnb
        Linear4bit = bnb.nn.Linear4bit
        if not _U._USE_NF4_KERNELS or not _wanted():
            return False
    except Exception:
        return False
    current = Linear4bit.forward
    if getattr(current, _MARK, False):
        return True
    global _BNB_OPS, _BNB_FUSED, _BNB_PICK, _BNB_FUSED_MAX_ROWS
    _BNB_OPS = _bnb_ops_traceable()
    _BNB_FUSED = _bnb_has_fused_gemm()
    _BNB_PICK, _BNB_FUSED_MAX_ROWS = _bnb_fused_heuristic() if _BNB_FUSED else (None, 1536)
    _FALLBACK["forward"] = current
    Linear4bit.forward = _nf4_forward
    return True


def uninstall_bnb_nf4_override():
    try:
        import bitsandbytes as bnb
    except Exception:
        return
    if getattr(bnb.nn.Linear4bit.forward, _MARK, False) and "forward" in _FALLBACK:
        bnb.nn.Linear4bit.forward = _FALLBACK["forward"]
