# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image speed levers: the MiniMax-H3 video ones, engaged through the structural gates of
``video_minimax_h3_attn`` (the vendored H3-Image attention has the same layout). ``UNSLOTH_H3_IMAGE_SPEED=0`` keeps
every one of them off; each also honours its video switch.

Exact (bit-identical outputs):
* ``h3_int8_gemm``: Studio's fused-dequant int8 GEMM on the resident INT8 DiT (UNSLOTH_DIFFUSION_INT8_GEMM=0); the row
  pads around the Linears it serves are dropped (its kernel takes any row count, each row is its own);
* ``h3_fused_qkv``: one row-concatenated int8 ``to_qkv`` per block, resident INT8 only, self-checked bit for bit
  (UNSLOTH_H3_FUSED_QKV=0);
* ``h3_strided_attn``: SDPA on permuted views, no copies (UNSLOTH_H3_STRIDED_ATTN=0);
* ``h3_shared_qkv_quant``: unfused Q / K / V share one activation quantization (UNSLOTH_H3_SHARED_QKV_QUANT=0); only where
  QKV is not fused, so it engages when UNSLOTH_H3_FUSED_QKV=0;
* the attention pin itself (``attention=_native_flash`` on sm80 / sm89): torch's default SDPA dispatch already picks
  flash there for bf16 head_dim 128, so the pin only fixes the kernel the strided path calls.
Tolerance-level:
* ``h3_qk_rope``: fused q/k RMSNorm + RoPE (Triton), rstd may differ in the last fp32 bit (UNSLOTH_H3_QK_ROPE=0);
* ``h3_fused_swiglu``: the down projection's activation quantization reads silu(gate) * value from the projection
  halves (not bit-exact) (UNSLOTH_H3_FUSED_SWIGLU=0);
* ``UNSLOTH_H3_ATTN_ARCH=0`` keeps the automatic cuDNN pick instead of flash on sm80 / sm89 (other arithmetic);
* ``h3_compiled``: regional torch.compile of the blocks, resident tiers only, opt-in (UNSLOTH_H3_IMAGE_COMPILE=1):
  its workspace is unmeasured beside the A100-40 dit_resident peak (37.5 of 40 GB), and a new packed sequence length
  recompiles once before dynamo marks it dynamic.
The streamed tier gets the attention levers only: its blocks move per forward, so nothing that needs hook-free,
resident projections (int8 GEMM swap, fused QKV, fused SwiGLU) engages there."""

from __future__ import annotations

import os
import types
from typing import Any, Optional

SPEED_ENV = "UNSLOTH_H3_IMAGE_SPEED"
COMPILE_ENV = "UNSLOTH_H3_IMAGE_COMPILE"
LEVERS_ATTR = "_unsloth_h3_speed_levers"
_RESIDENT_PLACEMENTS = ("resident", "dit_resident")


def speed_enabled() -> bool:
    return (os.environ.get(SPEED_ENV) or "1").strip().lower() not in ("0", "off", "false", "no")


def compile_enabled() -> bool:
    return (os.environ.get(COMPILE_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


def _row_pads(transformer: Any) -> list:
    """(parent, leaf, wrapper) of every H3-Image row pad under the blocks."""
    out = []
    for parent in transformer.modules():
        for leaf, child in list(parent._modules.items()):
            if type(child).__name__ == "H3ImageRowPad" and "inner" in child._modules:
                out.append((parent, leaf, child))
    return out


def _unwrap(pads: list, keep: Any = None) -> int:
    n = 0
    for parent, leaf, wrapper in pads:
        inner = wrapper._modules["inner"]
        if keep is not None and not keep(inner):
            continue
        setattr(parent, leaf, inner)
        n += 1
    return n


def _fuse_qkv(transformer: Any, logger: Any) -> int:
    """fuse_h3_qkv_ needs the bare projections: unwrap the Q / K / V pads, fuse, pad the fused ``to_qkv`` (and any
    projection left unfused) again, so the stock int8 path keeps its supported row multiple."""
    from .diffusion_minimax_h3_image import wrap_row_multiple
    from .video_minimax_h3_attn import fuse_h3_qkv_

    pads = [p for p in _row_pads(transformer) if p[1] in ("to_q", "to_k", "to_v")]
    _unwrap(pads)
    try:
        fused = fuse_h3_qkv_(transformer, logger = logger)
    finally:
        for parent, leaf, wrapper in pads:
            if leaf in parent._modules:  # left unfused
                setattr(parent, leaf, wrapper)
    for parent, _leaf, _wrapper in pads:
        if getattr(parent, "fused_projections", False) and type(
            parent._modules.get("to_qkv")
        ).__name__ != ("H3ImageRowPad"):
            wrap_row_multiple(parent, ["to_qkv"])
    return fused


def _fused_gemm_serves(inner: Any) -> bool:
    from .diffusion_int8_gemm import _REC
    rec = inner.__dict__.get(_REC)
    return bool(rec is not None and rec[4])


def install_h3_image_speed(
    pipe: Any,
    plan: Any,
    device: Any,
    logger: Any = None,
) -> tuple:
    """Engage the levers on ``pipe.transformer`` after placement, before the first forward. Returns (and records on
    the pipe as ``_unsloth_h3_speed_levers``) the engaged names; never raises."""
    levers: list = []
    try:
        levers = _install(pipe, plan, device, logger)
    except Exception as exc:  # noqa: BLE001 - an optimisation: the stock path stays
        if logger is not None:
            logger.warning("diffusion.minimax_h3_image: speed levers skipped: %s", exc)
    out = tuple(levers)
    try:
        setattr(pipe, LEVERS_ATTR, out)
    except Exception:  # noqa: BLE001
        pass
    if logger is not None and out:
        logger.info("diffusion.minimax_h3_image: speed levers %s", ",".join(out))
    return out


def _install(pipe: Any, plan: Any, device: Any, logger: Any) -> list:
    import torch

    transformer = getattr(pipe, "transformer", None)
    if transformer is None or not speed_enabled():
        return []
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return []
    levers: list = []
    resident = getattr(plan, "placement", None) in _RESIDENT_PLACEMENTS
    if resident and _fuse_qkv(transformer, logger):
        levers.append("h3_fused_qkv")
    if resident:
        from . import diffusion_int8_gemm as int8_gemm
        if int8_gemm.install(transformer, logger):
            # its kernel takes any row count: the pads around the Linears it serves only cost a copy
            _unwrap(_row_pads(transformer), keep = _fused_gemm_serves)
            levers.append("h3_int8_gemm")
    from .diffusion_attention import apply_attention_backend, select_attention_backend
    from .video_minimax_h3_attn import (
        h3_attention_backend,
        install_strided_attention,
        shared_qkv_count,
    )

    from .diffusion_minimax_h3_image import _module_dtype

    # the block stack's dtype: the fp32 embedders and heads come first in the parameter order
    dtype = _module_dtype(transformer, skip_fp32 = True)
    target = types.SimpleNamespace(device = dev.type, dtype = dtype, torch_device = str(dev))
    backend = select_attention_backend(target, None, speed_active = True)
    try:
        backend = h3_attention_backend(backend, requested = None, target = target)
    except Exception:  # noqa: BLE001 - keep the generic pick
        pass
    engaged = apply_attention_backend(pipe, backend, logger = logger, target = target)
    if engaged:
        levers.append(f"attention={engaged}")
    if engaged in ("_native_cudnn", "_native_flash") and install_strided_attention(
        transformer, logger = logger
    ):
        levers.append("h3_strided_attn")
        from .video_minimax_h3_qknorm import qk_rope_enabled

        procs = [getattr(m, "processor", None) for m in transformer.modules()]
        if qk_rope_enabled() and any(getattr(p, "_unsloth_qk_rope", False) for p in procs):
            levers.append("h3_qk_rope")
        if shared_qkv_count(transformer):
            levers.append("h3_shared_qkv_quant")
    if "h3_int8_gemm" in levers:
        from .video_minimax_h3_ffn import install as install_ffn
        if install_ffn(transformer, device = dev, logger = logger):
            levers.append("h3_fused_swiglu")
    if resident and compile_enabled():
        compiled = 0
        for block in getattr(transformer, "transformer_blocks", ()):
            block.compile()
            compiled += 1
        if compiled:
            levers.append("h3_compiled")
    return levers


def speed_levers(pipe: Any) -> Optional[tuple]:
    return getattr(pipe, LEVERS_ATTR, None)
