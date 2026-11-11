# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp32 tables built in ``__init__`` (Qwen-Image-2.1 timestep ``freqs``, MiniMax-H3 RoPE ``inv_freq``) stay fp32.
``from_pretrained`` leaves them as built; ``from_single_file`` ends with ``model.to(torch_dtype)``, rounding them."""

from __future__ import annotations

import functools
from typing import Any, Optional

_MARK = "_unsloth_fp32_init_buffers"


def is_fp32_init_table(module: Any, name: str, buffer: Any) -> bool:
    import torch
    return (
        buffer is not None
        and buffer.dtype == torch.float32
        and name in getattr(module, "_non_persistent_buffers_set", ())
    )


def _rounded(model: Any) -> bool:
    import torch
    low = (torch.bfloat16, torch.float16)
    return any(
        b is not None and b.dtype in low and n in m._non_persistent_buffers_set
        for m in model.modules()
        for n, b in m._buffers.items()
    )


def restore_fp32_init_buffers(model: Any, logger: Any = None) -> list:
    """Rebuilds the fp32 tables a cast rounded; call before offload hooks, which hold their own tensor refs."""
    if not callable(getattr(type(model), "from_config", None)) or not _rounded(model):
        return []
    from accelerate import init_empty_weights

    with init_empty_weights():  # parameters on meta, buffers built for real
        fresh = type(model).from_config(model.config)
    fixed = []
    for module_name, module in fresh.named_modules():
        for name, buffer in module._buffers.items():
            if not is_fp32_init_table(module, name, buffer) or buffer.device.type == "meta":
                continue
            try:
                target = model.get_submodule(module_name) if module_name else model
            except AttributeError:
                continue
            old = target._buffers.get(name)
            if old is None or old.dtype == buffer.dtype or old.shape != buffer.shape:
                continue
            if not old.is_floating_point() or old.device.type == "meta":
                continue
            target._buffers[name] = buffer.to(old.device)
            fixed.append(f"{module_name}.{name}" if module_name else name)
    if fixed and logger is not None:
        logger.info(
            "diffusion.load: %d fp32 __init__ table(s) kept in fp32 (%s)",
            len(fixed),
            ", ".join(fixed[:4]),
        )
    return fixed


def install_single_file_fp32_buffers(logger: Optional[Any] = None) -> bool:
    """Wraps ``FromOriginalModelMixin.from_single_file`` (every model / pipeline single-file load) once."""
    try:
        from diffusers.loaders.single_file_model import FromOriginalModelMixin
    except Exception:  # noqa: BLE001
        return False
    current = FromOriginalModelMixin.__dict__.get("from_single_file")
    func = getattr(current, "__func__", None)
    if func is None:
        return False
    if getattr(func, _MARK, False):
        return True

    @functools.wraps(func)
    def from_single_file(cls, *args: Any, **kwargs: Any) -> Any:
        model = func(cls, *args, **kwargs)
        try:
            restore_fp32_init_buffers(model, logger)
        except Exception as exc:  # noqa: BLE001 - the as-loaded model still renders
            if logger is not None:
                logger.warning("diffusion.load: fp32 __init__ tables not restored: %s", exc)
        return model

    setattr(from_single_file, _MARK, True)
    FromOriginalModelMixin.from_single_file = classmethod(from_single_file)
    return True
