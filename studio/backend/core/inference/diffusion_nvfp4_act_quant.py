# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One-pass NVFP4 quantiser for torchao's ``NVFP4Tensor.to_nvfp4`` (two-level scales, swizzled block scales).

torchao quantises in plain torch ops; under Inductor that becomes a per-element kernel that recomputes the block
scale for every element and writes an unpacked uint8 tensor, then a second kernel packs nibbles and a third
swizzles the scales (78 + 2 x 12 us for one 4128 x 3840 Z-Image activation on B200, about 3x the NVFP4 GEMM it
feeds). This kernel does all three in one pass over the activation. It is a ``triton_op``, so Inductor and CUDA
graph capture see an ordinary kernel.

Arithmetic is torchao's, in torchao's order (block max / 6, divide by the per-tensor scale, clamp to
[2^-6, 448], e4m3, then ``x * ((1 / s) / scale_fp8)``, clamp to +-6, round-to-nearest-even onto e2m1 by the
same bit trick), so the packed bytes and scales are bit-identical to torchao's own path; ``verify()`` checks
that on the device before the patch engages. ``UNSLOTH_NVFP4_FAST_ACT_QUANT=0`` restores torchao.
"""

from __future__ import annotations

import os
import threading
from typing import Any

FAST_ACT_QUANT_ENV = "UNSLOTH_NVFP4_FAST_ACT_QUANT"

_LOCK = threading.Lock()
_STATE: dict = {"installed": False, "verified": None, "reason": "", "fallback": 0}
_OP = None
# Tile shape (rows x 16*g elements per program): the fastest bit-identical one of 48 swept on B200.
_TILE = {"rows": 8, "g": 16, "warps": 2}
_ORIGINAL = None


try:  # Module scope, not a closure: Inductor re-imports a user kernel and its helpers by their global names.
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - no Triton means torchao's quantiser
    triton = None

if triton is not None:

    @triton.jit
    def _e2m1_code(y):
        # Clamp to +-6, then e2m1 round-to-nearest-even by the bit trick torchao's f32 -> f4 lowering emits.
        y = tl.minimum(tl.maximum(y, -6.0), 6.0)
        bits = y.to(tl.int32, bitcast = True)
        sign = bits & -2147483648
        mag_bits = bits ^ sign
        mag = mag_bits.to(tl.float32, bitcast = True)
        sat = mag >= 6.0
        denorm = (sat == 0) & (mag < 1.0)
        normal = (sat | denorm) == 0
        n_code = ((mag_bits + -1054867457 + ((mag_bits >> 22) & 1)) >> 22).to(tl.uint8)
        d_code = ((mag + 4194304.0).to(tl.int32, bitcast = True) - 1249902592).to(tl.uint8)
        code = tl.where(normal, n_code, tl.where(denorm, d_code, 7).to(tl.uint8))
        return code | ((sign >> 28).to(tl.uint8) & 8)

    @triton.jit
    def _nvfp4_act_quant_kernel(
        x_ptr,
        s_ptr,
        q_ptr,
        sf_ptr,
        M,
        K: tl.constexpr,
        NCB: tl.constexpr,
        ROWS: tl.constexpr,
        G: tl.constexpr,
    ):
        # One program: ROWS rows x G blocks of 16 elements, held as [ROWS, G, 8] even / odd halves so each packed
        # byte is (even | odd << 4) with no reshape or split (layout conversions some tile shapes miscompiled).
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * ROWS + tl.arange(0, ROWS)
        blocks = pid_k * G + tl.arange(0, G)
        lane = tl.arange(0, 8)
        rmask = (rows < M)[:, None, None]
        base = rows[:, None, None].to(tl.int64) * K + blocks[None, :, None] * 16 + lane[None, None, :] * 2
        xe = tl.load(x_ptr + base, mask = rmask, other = 0.0).to(tl.float32)
        xo = tl.load(x_ptr + base + 1, mask = rmask, other = 0.0).to(tl.float32)
        s = tl.load(s_ptr).to(tl.float32)
        bmax = tl.maximum(tl.max(tl.abs(xe), axis = 2), tl.max(tl.abs(xo), axis = 2))
        # torchao: block_scale = max_abs / 6 (Inductor folds it to * (1/6)); scaled = block_scale / s.
        bscale = (bmax * 0.16666666666666666) / s
        bscale = tl.minimum(tl.maximum(bscale, 0.015625), 448.0)
        sf8 = bscale.to(tl.float8e4nv)
        sf32 = sf8.to(tl.float32)
        recip = (((1.0 / s) * 1.0) / sf32)[:, :, None]
        lo = _e2m1_code(xe * recip)
        hi = _e2m1_code(xo * recip)
        packed = lo | (hi << 4)
        qoffs = rows[:, None, None].to(tl.int64) * (K // 2) + blocks[None, :, None] * 8 + lane[None, None, :]
        tl.store(q_ptr + qoffs, packed, mask = rmask)
        # cuBLAS / torchao ``to_blocked`` layout: 128 x 4 tiles, (row % 32) * 16 + (row // 32 % 4) * 4 + col % 4.
        rr = rows % 128
        tile = (rows // 128)[:, None] * NCB + (blocks // 4)[None, :]
        soffs = tile.to(tl.int64) * 512 + ((rr % 32) * 16 + (rr // 32) * 4)[:, None] + (blocks % 4)[None, :]
        tl.store(sf_ptr + soffs, sf8.to(tl.uint8, bitcast = True), mask = (rows < M)[:, None])


def fast_act_quant_enabled() -> bool:
    return os.environ.get(FAST_ACT_QUANT_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def describe() -> dict:
    return dict(_STATE)


def _build_op():
    """Register ``unsloth_nvfp4::act_quant(x, per_tensor_scale) -> (qdata u8 [M, K/2], scales u8 flat)``."""
    global _OP
    if _OP is not None:
        return _OP
    import torch
    from torch.library import triton_op, wrap_triton

    if triton is None:
        raise RuntimeError("triton unavailable")

    # ``from __future__ import annotations`` leaves the op's annotations as strings that infer_schema resolves here.
    globals().setdefault("torch", torch)

    @triton_op("unsloth_nvfp4::act_quant", mutates_args = ())
    def act_quant(x: torch.Tensor, per_tensor_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m, k = x.shape
        kb = k // 16
        ncb = (kb + 3) // 4
        m_pad = (m + 127) // 128 * 128
        q = torch.empty((m, k // 2), device = x.device, dtype = torch.uint8)
        numel = m_pad * ncb * 4
        if m_pad != m or ncb * 4 != kb:
            sf = torch.zeros((numel,), device = x.device, dtype = torch.uint8)
        else:
            sf = torch.empty((numel,), device = x.device, dtype = torch.uint8)
        g = _TILE["g"]
        while kb % g:
            g //= 2
        rows = _TILE["rows"]
        grid = (triton.cdiv(m, rows), kb // g)
        wrap_triton(_nvfp4_act_quant_kernel)[grid](
            x, per_tensor_scale, q, sf, m, K = k, NCB = ncb, ROWS = rows, G = g, num_warps = _TILE["warps"]
        )
        return q, sf

    _OP = act_quant
    return _OP


def _eligible(data_hp: Any, block_size: int, per_tensor_scale: Any, is_swizzled: bool, use_triton: bool) -> bool:
    import torch
    return (
        is_swizzled
        and not use_triton
        and block_size == 16
        and per_tensor_scale is not None
        and isinstance(per_tensor_scale, torch.Tensor)
        and per_tensor_scale.numel() == 1
        and data_hp.dim() == 2
        and data_hp.dtype == torch.bfloat16
        and data_hp.is_cuda
        and data_hp.shape[-1] % 16 == 0
        and data_hp.is_contiguous()
    )


def _traced(data_hp: Any) -> bool:
    """Whether this call is being traced into a graph rather than run eagerly.

    ``torch.compiler.is_compiling()`` alone is not enough: torchao's NVFP4 linear reaches ``to_nvfp4`` from the
    tensor subclass's dispatch, which AOTAutograd traces after Dynamo has finished, with that flag False. The
    tensors there are fake, so a fake input (or an active tracing context) marks the call as traced."""
    import torch

    if torch.compiler.is_compiling():
        return True
    try:
        from torch._subclasses.fake_tensor import is_fake

        if is_fake(data_hp):
            return True
    except Exception:  # noqa: BLE001 - older torch: fall through to the tracing context
        pass
    try:
        from torch._guards import TracingContext

        return TracingContext.try_get() is not None
    except Exception:  # noqa: BLE001
        return False


def _patched_to_nvfp4(
    data_hp,
    block_size: int = 16,
    per_tensor_scale = None,
    act_per_tensor_scale = None,
    is_swizzled_scales: bool = False,
    use_triton_kernel: bool = False,
    act_quant_kwargs = None,
):
    # Nothing mutable is read or written while Dynamo traces: a guard on a counter recompiles every call. The kill
    # switch is read once, at install; eager calls are counted (as torchao fallbacks), compiled ones show up as the
    # kernel in a profile.
    import torch

    counting = not _traced(data_hp)
    # Eager calls (the load-time weight pass, speed tier off) keep torchao's own eager arithmetic, which divides by 6
    # where Inductor multiplies by 1/6: the kernel matches the COMPILED path bit for bit, so only traced calls use it.
    if counting or not _eligible(data_hp, block_size, per_tensor_scale, is_swizzled_scales, use_triton_kernel):
        if counting:
            _STATE["fallback"] += 1
        return _ORIGINAL(
            data_hp,
            block_size,
            per_tensor_scale,
            act_per_tensor_scale,
            is_swizzled_scales,
            use_triton_kernel,
            act_quant_kwargs,
        )
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    from torchao.prototype.mx_formats.utils import hp_data_dims_to_swizzled_scale_dims_nvfp4

    m, k = data_hp.shape
    # A Triton launch lands on the CURRENT device, not the tensor's: guard it like every other NVFP4 launch.
    with torch.cuda.device(data_hp.device):
        q, sf = torch.ops.unsloth_nvfp4.act_quant(
            data_hp, per_tensor_scale.reshape(1).to(torch.float32)
        )
    scale_m, scale_k = hp_data_dims_to_swizzled_scale_dims_nvfp4(m, k)
    sf = sf.view(torch.float8_e4m3fn).view(scale_m, scale_k)
    return NVFP4Tensor(
        q,
        sf,
        block_size,
        data_hp.dtype,
        per_tensor_scale,
        act_per_tensor_scale,
        is_swizzled_scales,
        use_triton_kernel,
        act_quant_kwargs,
    )


def verify(device: Any = None) -> tuple:
    """Bit-identity against torchao's ``to_nvfp4`` as Inductor compiles it (the path Studio runs), on shapes with
    and without scale padding. Eager torchao divides by 6 where Inductor multiplies by 1/6, so eager can differ in
    the last fp32 bit of a block scale; that rate is recorded, not gated."""
    import torch
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

    original = _ORIGINAL or NVFP4Tensor.to_nvfp4
    op = _build_op()
    device = device or torch.device("cuda", torch.cuda.current_device())

    def reference(x):
        s = per_tensor_amax_to_scale(torch.max(torch.abs(x)))
        t = original(x, 16, s, None, True, False, None)
        return t.qdata.view(torch.uint8), t.scale.reshape(-1).view(torch.uint8), s

    compiled = torch.compile(reference, fullgraph = True, dynamic = False)
    op_compiled = torch.compile(lambda x, s: op(x, s), fullgraph = True, dynamic = False)
    with torch.cuda.device(device):
        return _verify_shapes(device, reference, compiled, op, op_compiled)


def _verify_shapes(device, reference, compiled, op, op_compiled) -> tuple:
    import torch

    gen = torch.Generator(device = device).manual_seed(0)
    eager_diff = 0
    for m, k in ((256, 512), (77, 3840), (300, 2560), (1, 64)):
        x = torch.randn(m, k, device = device, dtype = torch.float32, generator = gen)
        x = (x * torch.rand(m, 1, device = device, generator = gen) * 8).to(torch.bfloat16)
        x[0, :16] = 0
        x[-1, 16:32] = -1e-6
        want_q, want_sf, s = compiled(x)
        with torch.cuda.device(x.device):
            q, sf = op(x, s.reshape(1).to(torch.float32))
        if not torch.equal(want_q, q):
            return False, f"packed data differs from compiled torchao at {m}x{k}"
        if not torch.equal(want_sf, sf):
            return False, f"block scales differ from compiled torchao at {m}x{k}"
        # And as Inductor will run it: the op inside a compiled graph (catches a kernel Inductor cannot re-import).
        cq, csf = op_compiled(x, s.reshape(1).to(torch.float32))
        if not (torch.equal(cq, q) and torch.equal(csf, sf)):
            return False, f"the compiled op differs from the eager op at {m}x{k}"
        eq, esf, _ = reference(x)
        eager_diff += int((eq != q).sum().item()) + int((esf != sf).sum().item())
    _STATE["eager_bytes_differing"] = eager_diff
    return True, "bit-identical to compiled torchao to_nvfp4"


def install_nvfp4_act_quant_patch(logger: Any = None) -> bool:
    """Swap ``NVFP4Tensor.to_nvfp4`` for the one-pass kernel after an on-device bit-identity check. Idempotent."""
    global _ORIGINAL
    if not fast_act_quant_enabled():
        _STATE["reason"] = f"{FAST_ACT_QUANT_ENV}=0"
        return False
    with _LOCK:
        if _STATE["installed"]:
            return bool(_STATE["verified"])
        try:
            import torch
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

            if not torch.cuda.is_available():
                _STATE["reason"] = "cuda unavailable"
                return False
            _ORIGINAL = NVFP4Tensor.to_nvfp4
            ok, reason = verify()
        except Exception as exc:  # noqa: BLE001 - any failure keeps torchao's quantiser
            ok, reason = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        _STATE["verified"] = ok
        _STATE["reason"] = reason
        if ok:
            NVFP4Tensor.to_nvfp4 = staticmethod(_patched_to_nvfp4)
            _STATE["installed"] = True
    if logger is not None:
        try:
            logger.info(f"[nvfp4] one-pass activation quantiser: {'on' if ok else 'off'} ({reason})")
        except Exception:  # noqa: BLE001
            pass
    return ok


def uninstall_nvfp4_act_quant_patch() -> None:
    with _LOCK:
        if _STATE["installed"] and _ORIGINAL is not None:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
            NVFP4Tensor.to_nvfp4 = staticmethod(_ORIGINAL)
        _STATE["installed"] = False
        _STATE["verified"] = None
