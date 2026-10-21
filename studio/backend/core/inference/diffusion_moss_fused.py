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

Kill switch: ``UNSLOTH_DIFFUSION_MOSS_FUSED=0`` (``UNSLOTH_DIFFUSION_MOSS_FUSED_QKV=0`` keeps only the int8 runner).
"""

from __future__ import annotations

import os
import threading
from typing import Any

MOSS_FUSED_ENV = "UNSLOTH_DIFFUSION_MOSS_FUSED"
MOSS_FUSED_QKV_ENV = "UNSLOTH_DIFFUSION_MOSS_FUSED_QKV"
_LOCK = threading.Lock()


def _off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("0", "off", "false", "no")


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
    result = {"fused_qkv": 0, "int8_runner": 0}
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
    from .diffusion_moss_image import FUSED_QKV_ATTR, LINEAR_RUNNER_ATTR
    from .diffusion_zimage_fused import _fuse_linears, _share_storage

    result = {"fused_qkv": 0, "int8_runner": 0}
    dev = resident_cuda_device(root)
    if dev is None or moss_fused_disabled():
        return result
    blocks = _blocks(root)
    int8 = any(_int8_runnable(b.proj["q"]) for b in blocks)
    armed = arm_linear(dev) if int8 else False
    fuse = not _off(MOSS_FUSED_QKV_ENV)
    with _LOCK:
        for block in blocks:
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
            "diffusion.moss_fused: one q/k/v GEMM on %d of %d block(s); int8 runner on %d (act quant kernel %s)",
            result["fused_qkv"],
            len(blocks),
            result["int8_runner"],
            "probed" if armed else "torchao's",
        )
    return result


def uninstall(transformer: Any = None) -> None:
    """Drop the fused Linears (the per-row views stay valid) and any pending install."""
    if transformer is None:
        return
    from .diffusion_int8_fused import cancel_first_call
    from .diffusion_moss_image import FUSED_QKV_ATTR, LINEAR_RUNNER_ATTR

    cancel_first_call(transformer, "moss_fused")
    try:
        blocks = _blocks(transformer)
    except Exception:  # noqa: BLE001
        return
    with _LOCK:
        for block in blocks:
            block.__dict__.pop(FUSED_QKV_ATTR, None)
            block.__dict__.pop(LINEAR_RUNNER_ATTR, None)
