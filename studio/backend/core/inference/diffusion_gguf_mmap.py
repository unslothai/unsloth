# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF reads on the diffusers path without the whole-file host copy.

diffusers' ``load_gguf_checkpoint`` copies every tensor out of the reader's memory map
(``torch.from_numpy(tensor.data.copy())``), so the whole file sits in anonymous host memory until the weights reach
the device: 3.5 GiB of a 5.6 GiB load peak for FLUX.3 klein Q4_K_M on an L4. The mapped loader returns the same
tensors (same bytes, dtypes, shapes, types, quant types) as views of a private copy-on-write mapping, so untouched
pages stay page cache the kernel can reclaim, and a write (none on the load path) copies only its own page.
UNSLOTH_DIFFUSION_GGUF_MMAP=0 restores the copy. Off on Windows, where a mapped file cannot be deleted or replaced.
"""

from __future__ import annotations

import inspect
import os
import threading
from typing import Any

GGUF_MMAP_ENV = "UNSLOTH_DIFFUSION_GGUF_MMAP"
# The stock body this replaces; any other body (upstream changed it) keeps the stock function.
_STOCK_COPY_LINE = "torch.from_numpy(tensor.data.copy())"
_LOCK = threading.Lock()
_STATE: dict = {}
# Files read through the mapped loader in this process (engagement evidence for tests and logs).
MAPPED_LOADS: list = []


def gguf_mmap_enabled() -> bool:
    if os.name == "nt":
        return False
    return os.environ.get(GGUF_MMAP_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def _stock_is_copying(fn: Any) -> bool:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return _STOCK_COPY_LINE in source


def _mapped_load_gguf_checkpoint(gguf_checkpoint_path, return_tensors = False):
    """``load_gguf_checkpoint`` with tensors viewing a copy-on-write mapping instead of copied out of it."""
    stock = _STATE.get("stock")
    if not gguf_mmap_enabled() and stock is not None:
        return stock(gguf_checkpoint_path, return_tensors)
    import gguf
    import torch
    from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    from gguf import GGUFReader

    # "c": MAP_PRIVATE, writable views (torch never sees a read-only buffer), the file itself never written.
    reader = GGUFReader(gguf_checkpoint_path, "c")
    unquantized = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    parsed_parameters = {}
    nbytes = 0
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type
        is_gguf_quant = quant_type not in unquantized
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: "
                    "https://github.com/huggingface/diffusers"
                )
            )
        weights = torch.from_numpy(tensor.data)
        nbytes += int(tensor.data.nbytes)
        parsed_parameters[name] = (
            GGUFParameter(weights, quant_type = quant_type) if is_gguf_quant else weights
        )
    MAPPED_LOADS.append((str(gguf_checkpoint_path), nbytes))
    return parsed_parameters


def install_gguf_mmap(logger: Any = None) -> bool:
    """Route diffusers' GGUF reads through the mapped loader; idempotent. False when off or not applicable."""
    if not gguf_mmap_enabled():
        return False
    try:
        from diffusers.models import model_loading_utils as mlu
    except Exception:  # noqa: BLE001 - no diffusers GGUF support: nothing to replace
        return False
    with _LOCK:
        current = getattr(mlu, "load_gguf_checkpoint", None)
        if current is _mapped_load_gguf_checkpoint:
            return True
        if current is None or not _stock_is_copying(current):
            return False
        _STATE["stock"] = current
        mlu.load_gguf_checkpoint = _mapped_load_gguf_checkpoint
    if logger is not None:
        try:
            logger.info(
                "diffusion.gguf: GGUF tensors read as views of the file mapping (no whole-file host copy)"
            )
        except Exception:  # noqa: BLE001
            pass
    return True


def uninstall_gguf_mmap() -> None:
    """Put the stock loader back. Idempotent."""
    try:
        from diffusers.models import model_loading_utils as mlu
    except Exception:  # noqa: BLE001
        _STATE.clear()
        return
    with _LOCK:
        stock = _STATE.pop("stock", None)
        if stock is not None and mlu.load_gguf_checkpoint is _mapped_load_gguf_checkpoint:
            mlu.load_gguf_checkpoint = stock
