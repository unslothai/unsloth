# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image RoPE in real arithmetic (Inductor cannot lower the complex multiply): points
``ROPE_PER_DEVICE["cuda"]`` at ``diffusion_qwenimage21_rope``'s probed fma wrapper, bit-identical to the complex product.
Kill switch: ``UNSLOTH_DIFFUSION_QWEN_REAL_ROPE=0`` (``UNSLOTH_DIFFUSION_Q21_REAL_ROPE=0`` also disables it).
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any

QWEN_REAL_ROPE_ENV = "UNSLOTH_DIFFUSION_QWEN_REAL_ROPE"
_MODULE = "diffusers.models.transformers.transformer_qwenimage"
_FINGERPRINT = frozenset({"ba81dd3fc907c23a"})  # apply_rotary_emb_qwen, shared with Qwen-Image-2.1
_LOCK = threading.Lock()
_STOCK: dict = {}


def disabled() -> bool:
    from .diffusion_qwenimage21_rope import real_rope_disabled
    return (os.environ.get(QWEN_REAL_ROPE_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ) or real_rope_disabled()


def _patch_table(index: int, logger: Any = None) -> bool:
    import importlib

    import torch

    from . import diffusion_qwenimage21_rope as q21

    if not q21.inductor_addcmul_is_fma():
        return False
    mod = importlib.import_module(_MODULE)
    table = getattr(mod, "ROPE_PER_DEVICE", None)
    entry = table.get("cuda") if isinstance(table, dict) else None
    if entry is None:
        return False
    with _LOCK:
        stock = getattr(entry, "func", None)
        patched = getattr(stock, "__unsloth_real_rope__", False)
        if not patched and (
            stock is None
            or dict(getattr(entry, "keywords", {}) or {}) != {"use_real": False}
            or q21._digest(stock) not in _FINGERPRINT
        ):
            if logger is not None:
                logger.info("diffusion.qwenimage_rope: complex RoPE kept: RoPE table entry differs")
            return False
        if index not in q21._FUSION:
            q21._FUSION[index] = q21.probe_fusion(torch.device("cuda", index))
        if q21._FUSION[index] is None:
            return False
        if patched:
            return True
        q21._NEEDS_EMULATE[0] = q21._addcmul_lowering()[1]
        wrapper = q21._WRAPPERS.get(stock)
        if wrapper is None:
            wrapper = q21._WRAPPERS[stock] = q21._make_rope(stock)
        _STOCK[id(table)] = (table, entry)
        table["cuda"] = functools.partial(wrapper, use_real = False)
    if logger is not None:
        logger.info(
            "diffusion.qwenimage_rope: RoPE runs in real arithmetic inside the compiled blocks"
        )
    return True


def install(transformer: Any, logger: Any = None) -> bool:
    """Before the first compile; patches at the first forward if the DiT is not yet on the GPU."""
    if disabled():
        uninstall()  # a previous load may have patched the process-global table
        return False
    if type(transformer).__name__ != "QwenImageTransformer2DModel":
        return False
    from .diffusion_int8_fused import resident_cuda_device, run_on_first_call

    def _finish(t: Any) -> bool:
        dev = resident_cuda_device(t)
        if dev is None:
            return False
        import torch

        return _patch_table(
            dev.index if dev.index is not None else torch.cuda.current_device(), logger
        )

    if resident_cuda_device(transformer) is not None:
        return _finish(transformer)
    run_on_first_call(transformer, "qwenimage_rope", _finish)
    return True


def uninstall(transformer: Any = None) -> None:
    """Restore the stock table entry; drop ``transformer``'s pending first-forward install."""
    if transformer is not None:
        from .diffusion_int8_fused import cancel_first_call
        cancel_first_call(transformer, "qwenimage_rope")
    with _LOCK:
        for table, entry in list(_STOCK.values()):
            table["cuda"] = entry
        _STOCK.clear()
