# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistent prompt-conditioning cache for the diffusion INFERENCE path.

The inference sibling of the trainers' ``cond_cache_dir`` (see
``diffusion_train_common.DiffusionLoraConfig``), reusing the SAME on-disk store
(``diffusion_train_extras.PersistentConditioningCache``): one safetensors file
per encoded prompt, keyed by content hash + family. Enabled by pointing
``UNSLOTH_DIFFUSION_COND_CACHE_DIR`` at a directory -- unset/blank means off,
exactly like the trainer knob, so the default render path stays untouched.

When enabled, the loaded pipeline's ``encode_prompt`` is wrapped per-instance:
a repeated prompt returns the stored embeddings and never runs the text-encoder
forward, so under an offload policy the multi-GB encoders stay OFF the GPU for
warm prompts entirely. Verified bit-identical on the 32-image eval suites
(FLUX.1-schnell / FLUX.2-klein / Qwen-Image-2512): renders from cached
embeddings match inline-encoded renders exactly, and warm runs load the whole
pipeline with no text encoder resident (8-18s load, ~7B-VL VRAM saved on
Qwen-Image).

Safety gates: the wrapper only caches calls whose arguments are all plain
JSON-safe values (a tensor argument such as pre-supplied ``prompt_embeds``
passes straight through), keys on everything that changes the embedding
numerics (family, repo, dtype, text-encoder quant, diffusers version, and the
full argument set minus device/generator), and bypasses entirely while LoRA
adapters are attached (an adapter may target the text encoders). Best-effort
throughout: any cache failure falls back to the real encode. torch is imported
lazily so this stays importable in a no-torch runtime.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from typing import Any, Optional

_ENV_DIR = "UNSLOTH_DIFFUSION_COND_CACHE_DIR"

# Bound arguments that never change the returned embeddings: the target device
# is a placement detail (the hit is moved there) and no encode path draws RNG.
_KEY_EXCLUDED_ARGS = frozenset({"device", "generator"})


def cache_dir() -> Optional[str]:
    """The configured cache directory, or None when the cache is off. A blank
    value means off (matching the trainers' ``cond_cache_dir`` semantics)."""
    value = (os.environ.get(_ENV_DIR) or "").strip()
    return value or None


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    return False


# Per-slot layout codes for _flatten/_unflatten (stored as the leading int64 tensor):
# -1 = None slot, -2 = a bare tensor, n >= 0 = a LIST of n tensors (Z-Image returns
# its per-prompt embeddings as a list, so one nesting level must round-trip).
_SLOT_NONE = -1
_SLOT_TENSOR = -2


def _flatten(result: Any) -> Optional[list]:
    """Flatten an ``encode_prompt`` result tuple of tensors / None / tensor-lists
    into ``[layout_tensor, *tensors]`` for the conditioning cache, or None when the
    result holds anything else (those calls simply aren't cached)."""
    if not isinstance(result, tuple) or not result:
        return None
    layout: list[int] = []
    flat: list[Any] = []
    for item in result:
        if item is None:
            layout.append(_SLOT_NONE)
        elif isinstance(item, (list, tuple)):
            if not all(hasattr(t, "detach") for t in item):
                return None
            layout.append(len(item))
            flat.extend(item)
        elif hasattr(item, "detach"):
            layout.append(_SLOT_TENSOR)
            flat.append(item)
        else:
            return None
    import torch  # noqa: PLC0415

    return [torch.tensor(layout, dtype = torch.int64), *flat]


def _unflatten(stored: tuple, device: Any) -> tuple:
    """Rebuild the ``encode_prompt`` result from a cached ``_flatten`` record,
    moving every tensor to ``device`` (dtype is preserved as stored)."""
    layout = stored[0].tolist()
    tensors = list(stored[1:])
    out: list[Any] = []
    index = 0
    for code in layout:
        if code == _SLOT_NONE:
            out.append(None)
        elif code == _SLOT_TENSOR:
            out.append(tensors[index].to(device))
            index += 1
        else:
            out.append([t.to(device) for t in tensors[index : index + code]])
            index += code
    return tuple(out)


def install(
    pipe: Any,
    *,
    family: str,
    repo_id: str,
    dtype: Any,
    te_quant: Any = None,
    logger: Any = None,
) -> bool:
    """Wrap ``pipe.encode_prompt`` with the persistent cache. No-op (False) when
    the env knob is unset, the pipe has no ``encode_prompt``, or setup fails.

    Instance-level assignment only: the wrapper dies with the pipe on unload and
    never mutates the pipeline class."""
    root = cache_dir()
    if root is None:
        return False
    encode = getattr(pipe, "encode_prompt", None)
    if not callable(encode):
        return False
    try:
        # Lazy: core.training imports parts of core.inference, so the module-level
        # import would be circular; the extras module itself is stdlib-only.
        from core.training.diffusion_train_extras import PersistentConditioningCache

        signature = inspect.signature(encode)
        cache = PersistentConditioningCache(root, family, 0)
    except Exception as exc:  # noqa: BLE001 — cache is best-effort
        if logger is not None:
            logger.warning("diffusion.cond_cache: install failed: %s", exc)
        return False

    # Everything beyond the call arguments that changes the embedding numerics.
    load_fp = {
        "repo": str(repo_id),
        "dtype": str(dtype),
        "te_quant": str(te_quant) if te_quant is not None else "none",
        "diffusers": _diffusers_version(),
    }
    stats = {"hits": 0, "misses": 0}

    def cached_encode_prompt(*args: Any, **kwargs: Any) -> Any:
        try:
            # LoRA may target the text encoders; adapters attached -> always encode.
            if getattr(pipe, "_unsloth_loras", ()):
                return encode(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            keyed = {
                name: value
                for name, value in bound.arguments.items()
                if name not in _KEY_EXCLUDED_ARGS
            }
            if not all(_json_safe(v) for v in keyed.values()):
                # Tensor/object arguments (pre-supplied embeds, images) are not keyable.
                return encode(*args, **kwargs)
            payload = json.dumps(
                {"load": load_fp, "args": keyed}, sort_keys = True, default = str
            )
            key = cache.text_key(
                f"inference::{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
            )
            hit = cache.get(key)
            if hit is not None:
                stats["hits"] += 1
                return _unflatten(hit, _target_device(pipe, bound))
        except Exception as exc:  # noqa: BLE001 — never fail a generation over the cache
            if logger is not None:
                logger.warning("diffusion.cond_cache: lookup failed: %s", exc)
            return encode(*args, **kwargs)
        result = encode(*args, **kwargs)
        try:
            flat = _flatten(result)
            if flat is not None:
                cache.put(key, flat)
                stats["misses"] += 1
        except Exception as exc:  # noqa: BLE001 — a failed write only skips reuse
            if logger is not None:
                logger.warning("diffusion.cond_cache: store failed: %s", exc)
        return result

    pipe.encode_prompt = cached_encode_prompt
    pipe._unsloth_cond_cache_stats = stats
    if logger is not None:
        logger.info(
            "diffusion.cond_cache: enabled at %s (family=%s); repeated prompts skip "
            "the text-encoder forward",
            root,
            family,
        )
    return True


def _target_device(pipe: Any, bound: inspect.BoundArguments) -> Any:
    """The device a hit's tensors must land on: the caller's explicit ``device``
    argument when given, else the pipeline's execution device, else CPU."""
    device = bound.arguments.get("device")
    if device is not None:
        return device
    device = getattr(pipe, "_execution_device", None)
    return device if device is not None else "cpu"


def _diffusers_version() -> Optional[str]:
    try:
        import diffusers  # noqa: PLC0415

        return str(getattr(diffusers, "__version__", None))
    except Exception:  # noqa: BLE001
        return None
