# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-VL ``patch_embed.proj`` (Conv3d, kernel == stride) as one F.linear.

torch 2.9.x refuses cuDNN for bf16 / fp16 Conv3d on cuDNN 9.8-9.14 (``use_cudnn`` in
Convolution.cpp; fixed in 2.10), and the 2.9 wheels ship 9.10.2, so it hits slow_conv3d
(37 s for 4096 patches on an L4 vs under 1 ms as a matmul).
``UNSLOTH_PATCH_EMBED_LINEAR=0`` keeps the stock conv, ``=1`` forces it on any CUDA torch.
"""

from __future__ import annotations

import os
import re
import types
from typing import Any, Optional

PATCH_EMBED_LINEAR_ENV = "UNSLOTH_PATCH_EMBED_LINEAR"
_FALSE = ("0", "false", "no", "off")
_TRUE = ("1", "true", "yes", "on", "force")
_CUDNN_REFUSED = (90800, 91500)


def _torch_version(version: str) -> Optional[tuple[int, int, int]]:
    try:
        parts = str(version).split("+", 1)[0].split(".")
        m = re.match(r"\d+", parts[2] if len(parts) > 2 else "0")
        return int(parts[0]), int(parts[1]), int(m.group(0)) if m else 0
    except (IndexError, TypeError, ValueError):
        return None


def conv3d_skips_cudnn(
    torch_version: str,
    cudnn_version: Optional[int],
    compiled_cudnn: Optional[int] = None,
) -> bool:
    """True when this torch build runs half / bf16 Conv3d (kernel != 1) on the native kernel.
    2.9.0 checks its compile-time cuDNN with no upper bound; 2.9.1+ checks the runtime one in [9.8, 9.15)."""
    v = _torch_version(torch_version)
    if v is None or v[:2] != (2, 9):
        return False
    lo, hi = _CUDNN_REFUSED
    if v[2] == 0:
        ref = compiled_cudnn if compiled_cudnn is not None else cudnn_version
        return ref is not None and int(ref) >= lo
    return cudnn_version is not None and lo <= int(cudnn_version) < hi


def _compiled_cudnn(torch: Any) -> Optional[int]:
    try:
        major, minor, patch = torch._C._cudnn.getCompileVersion()
        return int(major) * 10000 + int(minor) * 100 + int(patch)
    except Exception:  # noqa: BLE001 - no compile-time cuDNN version: use the runtime one
        return None


def _mode() -> str:
    value = os.environ.get(PATCH_EMBED_LINEAR_ENV, "").strip().lower()
    if value in _FALSE:
        return "off"
    if value in _TRUE:
        return "force"
    return "auto"


def _runtime_affected() -> bool:
    import torch

    from .rocm_bf16 import is_rocm_torch

    if is_rocm_torch(torch) or not torch.cuda.is_available():
        return False
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001 - no cuDNN: nothing to work around
        return False
    return conv3d_skips_cudnn(torch.__version__, cudnn, _compiled_cudnn(torch))


def is_patchify_conv3d(conv: Any) -> bool:
    """A Conv3d that tiles its input: kernel == stride, no padding / dilation / groups."""
    import torch

    if not isinstance(conv, torch.nn.Conv3d) or type(conv).forward is not torch.nn.Conv3d.forward:
        return False
    kernel = tuple(conv.kernel_size)
    padding = conv.padding
    return (
        kernel == tuple(conv.stride)
        and any(k != 1 for k in kernel)
        and (padding == "valid" or tuple(padding) == (0, 0, 0))
        and tuple(conv.dilation) == (1, 1, 1)
        and conv.groups == 1
        and conv.padding_mode == "zeros"
        and isinstance(getattr(conv, "weight", None), torch.nn.Parameter)
    )


def patch_conv_as_linear(x: Any, weight: Any, bias: Any) -> Any:
    """``F.conv3d(x, weight, bias, stride = kernel)`` for x of shape [N, *weight.shape[1:]], as a matmul."""
    import torch.nn.functional as F

    out = F.linear(x.reshape(x.shape[0], -1), weight.reshape(weight.shape[0], -1), bias)
    return out.view(x.shape[0], weight.shape[0], 1, 1, 1)


def _is_cuda(x: Any) -> bool:
    return bool(x.is_cuda)


def _linear_patch_conv(module: Any, x: Any) -> Any:
    """One patch per row ([N, C, kt, kh, kw]) on CUDA in half / bf16: the matmul; anything else: stock."""
    import torch

    weight = module.weight
    if (
        x.dim() != 5
        or tuple(x.shape[1:]) != tuple(weight.shape[1:])
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.dtype != weight.dtype
        or not _is_cuda(x)
    ):
        return torch.nn.Conv3d.forward(module, x)
    return patch_conv_as_linear(x, weight, module.bias)


def install_patch_embed_linear(
    pipe: Any,
    target: Any,
    *,
    logger: Any = None,
    affected: Optional[bool] = None,
) -> int:
    """Swap every ``*PatchEmbed.proj`` patchify Conv3d in the pipeline for the matmul; returns the count."""
    mode = _mode()
    if mode == "off" or getattr(target, "device", None) != "cuda":
        return 0
    if mode == "auto":
        if affected is None:
            try:
                affected = _runtime_affected()
            except Exception:  # noqa: BLE001 - the stock conv stays
                affected = False
        if not affected:
            return 0
    patched = 0
    for component in getattr(pipe, "components", {}).values() or ():
        modules = getattr(component, "modules", None)
        if not callable(modules):
            continue
        for module in modules():
            if not type(module).__name__.endswith("PatchEmbed"):
                continue
            conv = getattr(module, "proj", None)
            if (
                conv is None
                or getattr(conv, "_unsloth_patch_embed_linear", False)
                or "forward" in conv.__dict__
                or not is_patchify_conv3d(conv)
            ):
                continue
            conv.forward = types.MethodType(_linear_patch_conv, conv)
            conv._unsloth_patch_embed_linear = True
            patched += 1
    if patched and logger is not None:
        logger.info(
            "diffusion.patch_embed_linear: %d vision patch embed(s) run as a matmul "
            "(torch 2.9 skips cuDNN for this Conv3d on cuDNN 9.8-9.14)",
            patched,
        )
    return patched
