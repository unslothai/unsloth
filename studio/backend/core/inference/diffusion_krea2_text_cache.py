# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 text fusion, text projection and rotary table run once per prompt, not once per step.

``Krea2Transformer2DModel.forward`` runs ``text_fusion`` (four text blocks over every token's 12 tapped encoder
layers), ``txt_in`` and ``rotary_emb`` on every denoise call, though none of them reads the timestep or the latents:
they see the same prompt embeddings, mask and position ids at every step of a render (the text trim hands the same
trimmed tensors back each step). Each module keeps its last outputs per input (a CFG pair is two entries) and returns
them while the same, unmodified input memory comes back: bit-identical. 9.4 of 186 ms per 1024^2 forward on G4.

Entries live for one render: Studio's generate clears them when the render ends. Within a render the inputs are
matched on identity / address and, where torch keeps one, the version counter (inference-mode tensors have none, so an
in-place edit of the prompt tensor in the middle of a render is not seen); the autocast dtype is part of the key.

Resident modules only: a streamed module keeps stock calls (skipping its forward would leave its offload group out
of the prefetch order). Calls under CUDA-graph capture or tracing run stock, so a graph never records a cached
output's address. ``UNSLOTH_DIFFUSION_KREA2_TEXT_CACHE=0`` runs them every step again.
"""

from __future__ import annotations

import os
from typing import Any, Optional

TEXT_CACHE_ENV = "UNSLOTH_DIFFUSION_KREA2_TEXT_CACHE"
CLASS_NAME = "Krea2Transformer2DModel"
MODULES = ("text_fusion", "txt_in", "rotary_emb")
_INSTALLED_ATTR = "_unsloth_krea2_text_cache"
_MEMO_ATTR = "_unsloth_krea2_text_cache_memo"
_MEMO_MAX = 4  # a CFG pair of this render, and the last render's
_OFF = ("0", "off", "false", "no")
STATS = {"hits": 0, "misses": 0, "stock": 0}


def enabled() -> bool:
    return (os.environ.get(TEXT_CACHE_ENV) or "").strip().lower() not in _OFF


def _version(t: Any) -> Any:
    # inference tensors have no version counter (Studio renders under inference_mode)
    return "inference" if t.is_inference() else t._version


def _sources(args: tuple, kwargs: dict) -> Optional[tuple]:
    """The call's inputs as (positional tensors, sorted keyword items), or None when one is not a tensor / None."""
    import torch

    flat = []
    for value in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if value is not None and not torch.is_tensor(value):
            return None
        flat.append(value)
    return tuple(flat), tuple(sorted(kwargs))


def _geometry(t: Any) -> tuple:
    root = t._base if t._base is not None else t
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, _version(root))


def _key(t: Any) -> tuple:
    """(the tensor itself, its address and geometry, version). The forward re-slices its mask every call
    (``mask[:, None, None, :]``): a new view object, and under inference_mode one with no ``_base``. Holding the
    earlier view keeps its storage alive, so an equal address, geometry and dtype is the same memory."""
    return (t, _geometry(t))


def _same(held: tuple, flat: tuple) -> bool:
    if len(held) != len(flat):
        return False
    for key, t in zip(held, flat):
        if key is None or t is None:
            if (key is None) != (t is None):
                return False
            continue
        if key[0] is not t and key[1] != _geometry(t):
            return False
        if key[0] is t and key[1][-1] != _version(t._base if t._base is not None else t):
            return False
    return True


def _streamed(module: Any) -> bool:
    for sub in module.modules():
        if getattr(sub, "_hf_hook", None) is not None:
            return True
        hooks = getattr(getattr(sub, "_diffusers_hook", None), "hooks", None) or {}
        if any("offload" in str(k).lower() for k in hooks):
            return True
    return False


def _amp_state(flat: tuple) -> Optional[tuple]:
    """Autocast state for the inputs' device (part of the key: an fp32 entry must not answer a bf16 autocast call);
    None when it cannot be read (then the call runs stock)."""
    import torch

    device = next((t.device.type for t in flat if t is not None), "cpu")
    try:
        on = bool(torch.is_autocast_enabled(device))
        return (device, on, torch.get_autocast_dtype(device) if on else None)
    except Exception:  # noqa: BLE001 - older torch: no per-device query
        return None


def _uncacheable() -> bool:
    import torch
    if torch.is_grad_enabled():
        return True
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return True


def _wrap(module: Any, stock: Any) -> Any:
    import torch

    def forward(*args: Any, **kwargs: Any) -> Any:
        if torch.compiler.is_compiling():
            # first, before any Python state is read or written: a counter or memo touched here becomes a Dynamo
            # guard that fails on every call (one recompile per step)
            return stock(*args, **kwargs)
        if not enabled() or _uncacheable() or _streamed(module):
            STATS["stock"] += 1
            return stock(*args, **kwargs)
        found = _sources(args, kwargs)
        if found is None:
            STATS["stock"] += 1
            return stock(*args, **kwargs)
        flat, names = found
        amp = _amp_state(flat)
        if amp is None:
            STATS["stock"] += 1
            return stock(*args, **kwargs)
        memo = module.__dict__.setdefault(_MEMO_ATTR, [])
        for held, held_names, held_amp, out in memo:
            if held_names == names and held_amp == amp and _same(held, flat):
                STATS["hits"] += 1
                return out
        out = stock(*args, **kwargs)
        held = tuple(None if t is None else _key(t) for t in flat)
        memo.insert(0, (held, names, amp, out))
        del memo[_MEMO_MAX:]
        STATS["misses"] += 1
        return out

    forward._unsloth_krea2_text_cache = True
    return forward


def installed(transformer: Any) -> bool:
    return bool(getattr(transformer, "__dict__", {}).get(_INSTALLED_ATTR))


def install(transformer: Any, logger: Any = None) -> bool:
    """Per transformer instance: wrap each module's forward (the instance attribute, inside any hooks it carries)."""
    if type(transformer).__name__ != CLASS_NAME or not enabled():
        return False
    if installed(transformer):
        return True
    for name in MODULES:
        if not hasattr(getattr(transformer, name, None), "forward"):
            return False
    for name in MODULES:
        module = getattr(transformer, name)
        stock = module.forward
        if getattr(stock, "_unsloth_krea2_text_cache", False):
            continue
        module.forward = _wrap(module, stock)
    transformer.__dict__[_INSTALLED_ATTR] = True
    if logger is not None:
        logger.info(
            "diffusion.krea2: text fusion, text projection and rotary table run once per prompt"
        )
    return True


def clear(transformer: Any) -> None:
    for name in MODULES:
        module = getattr(transformer, name, None)
        if module is not None:
            module.__dict__.pop(_MEMO_ATTR, None)


def end_render(pipe: Any) -> None:
    """Drop every entry when a render ends (Studio's generate ``finally``): an entry never outlives its request, so a
    later in-place edit of an inference-mode prompt tensor (no version counter) cannot be answered from the cache.
    Never raises."""
    try:
        transformer = getattr(pipe, "transformer", None)
        if installed(transformer):
            clear(transformer)
    except Exception:  # noqa: BLE001
        pass
