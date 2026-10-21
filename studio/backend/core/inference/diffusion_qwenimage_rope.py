# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image RoPE in real arithmetic (Inductor cannot lower the complex multiply): points the stock RoPE at
``diffusion_qwenimage21_rope``'s probed fma wrapper, bit-identical to the complex product. Two stock forms:
diffusers 0.41.0 routes through ``ROPE_PER_DEVICE["cuda"]`` (its entry is swapped); main d42ebf9c dropped the table
and the attention processor calls the module global ``apply_rotary_emb_qwen`` (the global is swapped, only while the
processor matches a known source). Kill switch: ``UNSLOTH_DIFFUSION_QWEN_REAL_ROPE=0``
(``UNSLOTH_DIFFUSION_Q21_REAL_ROPE=0`` also disables it).
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any

QWEN_REAL_ROPE_ENV = "UNSLOTH_DIFFUSION_QWEN_REAL_ROPE"
_MODULE = "diffusers.models.transformers.transformer_qwenimage"
_FINGERPRINT = frozenset({"ba81dd3fc907c23a"})  # apply_rotary_emb_qwen, shared with Qwen-Image-2.1
# QwenDoubleStreamAttnProcessor2_0.__call__ sources that call the module global apply_rotary_emb_qwen(..., use_real=False)
# directly (no ROPE_PER_DEVICE table): main d42ebf9c.
_PROCESSOR_FINGERPRINT = frozenset({"98406ca5a85151f8"})
_PROCESSOR = "QwenDoubleStreamAttnProcessor2_0"
_LOCK = threading.Lock()
_STOCK: dict = {}
_STOCK_GLOBAL: dict = {}  # id(module) -> (module, stock apply_rotary_emb_qwen), the table-less form


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
    if table is None:
        return _patch_global(mod, index, logger)
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
        # Per device, even once the table is patched: the wrapper keeps the complex path on an unprobed index.
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


def _probe(index: int) -> bool:
    import torch

    from . import diffusion_qwenimage21_rope as q21

    # Per device, even once patched: the wrapper keeps the complex path on an unprobed index.
    if index not in q21._FUSION:
        q21._FUSION[index] = q21.probe_fusion(torch.device("cuda", index))
    return q21._FUSION[index] is not None


def _patch_global(
    mod: Any,
    index: int,
    logger: Any = None,
) -> bool:
    """No table (main d42ebf9c): swap the module global the attention processor calls."""
    from . import diffusion_qwenimage21_rope as q21

    with _LOCK:
        current = getattr(mod, "apply_rotary_emb_qwen", None)
        patched = getattr(current, "__unsloth_real_rope__", False)
        if not patched:
            processor = getattr(mod, _PROCESSOR, None)
            call = vars(processor).get("__call__") if isinstance(processor, type) else None
            if (
                current is None
                or q21._digest(current) not in _FINGERPRINT
                or call is None
                or q21._digest(call) not in _PROCESSOR_FINGERPRINT
            ):
                if logger is not None:
                    logger.info(
                        "diffusion.qwenimage_rope: complex RoPE kept: RoPE call site differs"
                    )
                return False
        if not _probe(index):
            return False
        if patched:
            return True
        q21._NEEDS_EMULATE[0] = q21._addcmul_lowering()[1]
        wrapper = q21._WRAPPERS.get(current)
        if wrapper is None:
            wrapper = q21._WRAPPERS[current] = q21._make_rope(current)
        _STOCK_GLOBAL[id(mod)] = (mod, current)
        mod.apply_rotary_emb_qwen = wrapper
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
    from .diffusion_int8_fused import resident_cuda_cards, run_on_first_call

    def _finish(t: Any) -> bool:
        cards = resident_cuda_cards(t)
        passed = [index for index in cards if _probe(index)]
        return bool(passed) and _patch_table(passed[0], logger)

    if resident_cuda_cards(transformer):
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
        for mod, stock in list(_STOCK_GLOBAL.values()):
            mod.apply_rotary_emb_qwen = stock
        _STOCK_GLOBAL.clear()
